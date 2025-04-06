import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.optim
import logging
import os
import torch.distributed as dist

from modules.util.bf16_stochastic_rounding import addcdiv_stochastic_, add_stochastic_

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
    from prodigyopt import Prodigy  # <- Assumindo que Prodigy existe no escopo real
else:
    _params_t = Any
    Prodigy = None  # <- Mantido como no original para consistência


# --- step_prodigy (não modificado, parece correto) ---
@torch.no_grad()
def step_prodigy(self: Prodigy, closure=None):
    loss = closure() if closure is not None else None

    for group in self.param_groups:
        if group["lr"] == 0.0:
            continue

        beta1, beta2 = group["betas"]
        beta3 = group["beta3"] or math.sqrt(beta2)
        d, d_max, lr = group["d"], group["d_max"], group["lr"]
        use_bias_correction = group["use_bias_correction"]
        safeguard_warmup = group["safeguard_warmup"]
        fsdp_in_use = group["fsdp_in_use"]
        slice_p = group["slice_p"]
        decouple = group["decouple"]
        k = group["k"]

        bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1)) if use_bias_correction else 1.0
        dlr = d * lr * bias_correction

        group["d_numerator"] *= beta3
        d_numerator = group["d_numerator"]
        d_denom = 0.0

        # Vetorização: separa tensores válidos
        grads, params, exp_avgs, exp_avg_sqs, s_list, p0_list = [], [], [], [], [], []
        shapes_mismatch = False

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.data
            state = self.state.setdefault(p, {})
            if "step" not in state:
                state["step"] = 0
                # >>>>>>>>> INICIALIZAÇÃO ORIGINAL (SEM CPU OFFLOAD) <<<<<<<<<
                # Cria 's' e 'p0' no mesmo dispositivo de p
                state["s"] = torch.zeros_like(p.data.flatten()[::slice_p])
                state["p0"] = p.flatten()[::slice_p].clone() if p.any() else torch.tensor(0, device=p.device, dtype=p.dtype)
                if beta1 > 0:
                    # Cria no mesmo dispositivo de p, mas pode usar float32 para precisão
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                # Cria no mesmo dispositivo de p, mas pode usar float32 para precisão
                state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32)
                # >>>>>>>>> FIM DA INICIALIZAÇÃO ORIGINAL <<<<<<<<<

            # state refs
            s, p0 = state["s"], state["p0"]
            if lr > 0.0:
                sliced_grad = grad.flatten()[::slice_p]
                d_numerator += (d / group["d0"]) * dlr * torch.dot(sliced_grad, p0 - p.data.flatten()[::slice_p]).item()
                if safeguard_warmup:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / group["d0"]) * d))
                else:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / group["d0"]) * dlr))
                d_denom += s.abs().sum().item()

            # A partir daqui, preparamos vetorização se possível
            # Verifica se p não é bfloat16 ou se stochastic_rounding não está ativo
            if p.dtype != torch.bfloat16 or not getattr(self, "stochastic_rounding", False):
                # >>>>>>>>> Garante que os estados acessados aqui estão no dispositivo correto <<<<<<<<<
                # (A inicialização acima já garante isso)
                grads.append(grad)
                params.append(p.data)
                exp_avg_sqs.append(self.state[p]["exp_avg_sq"])
                if beta1 > 0:
                    exp_avgs.append(self.state[p]["exp_avg"])
            else:
                shapes_mismatch = True  # fallback para laço manual se for bf16 com rounding

        if d_denom == 0.0:
            group["k"] = k + 1
            continue

        # Sincronização distribuída se necessário
        if fsdp_in_use and grads:  # Só faz sentido sincronizar se houver gradientes/parâmetros
            # >>>>>>>>> Garante que o tensor para dist vai para o device correto <<<<<<<<<
            dist_device = params[0].device if params else (grads[0].device if grads else "cpu")  # Escolhe um device válido
            dist_tensor = torch.tensor([d_numerator, d_denom], device=dist_device)
            dist.all_reduce(dist_tensor)
            d_numerator, d_denom = dist_tensor.tolist()
            # Recalcula d_hat após all_reduce se d_denom não for zero
            if d_denom != 0.0:
                group["d_hat"] = group["d_coef"] * d_numerator / d_denom
            else:  # Evita divisão por zero
                group["d_hat"] = 0.0  # Ou algum outro valor padrão

        # Atualiza d_hat, d, d_max APÓS possível sincronização
        if d_denom != 0.0:  # Só atualiza se o denominador for válido
            d_hat = group["d_coef"] * d_numerator / d_denom
            if d == group["d0"]:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * group["growth_rate"])
            group["d_hat"] = d_hat  # Guarda o d_hat calculado

        # Salva novos valores
        group["d"] = d
        group["d_max"] = d_max
        # group['d_hat'] = d_hat # Já salvo acima
        group["d_numerator"] = d_numerator  # Salva o numerador (pós beta3 e pós all_reduce)
        group["d_denom"] = d_denom  # Salva o denominador (pós all_reduce)

        dlr = d * lr * bias_correction  # Recalcula após novo d

        # Atualizações vetorizadas se possível
        if grads and not shapes_mismatch:
            # Decoupled weight decay
            if group["weight_decay"] != 0 and decouple:
                torch._foreach_add_(params, params, alpha=-group["weight_decay"] * dlr)

            # EMA - Usa os estados que já estão no device correto
            # Convertendo gradientes para float32 para a atualização EMA vetorizada
            # >>>>>>>>> Conversão para Float32 no device original <<<<<<<<<
            grads_fp32 = torch._foreach_clone(grads, dtype=torch.float32)

            if beta1 > 0:
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads_fp32, alpha=d * (1 - beta1))  # Usa grads_fp32

            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads_fp32, grads_fp32, value=d * d * (1 - beta2))  # Usa grads_fp32

            del grads_fp32  # Libera memória

            # denom = sqrt(v) + eps*d - Operação feita no device dos estados
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_add_(denom, d * group["eps"])

            # Atualização final do parâmetro
            if beta1 > 0:
                torch._foreach_addcdiv_(params, exp_avgs, denom, value=-dlr)
            else:
                # Se beta1=0, usamos o gradiente original (não o fp32) multiplicado por d
                torch._foreach_addcdiv_(params, grads, denom, value=-dlr * d)

            # Avança step e limpa gradiente
            for p_ in group["params"]:  # Usar p_ para não conflitar com 'p' do loop anterior
                if p_.grad is not None:
                    # p_.grad = None # Gradiente já foi usado, não precisa limpar explicitamente aqui? torch._foreach_ o usa como input. Cuidado aqui.
                    self.state[p_]["step"] += 1
            # Limpeza explícita pode ser mais segura dependendo da interação com autograd
            for p_ in group["params"]:
                if p_.grad is not None:
                    p_.grad = None

        else:
            # Fallback (BF16 com stochastic_rounding ou shape mismatch)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # >>>>>>>>> Estados já estão no device correto <<<<<<<<<
                exp_avg_sq = state["exp_avg_sq"]

                # --- ATUALIZAÇÃO EMA PARA O FALLBACK ---
                # Converte gradiente para float32 no device original para precisão
                grad_fp32 = grad.to(dtype=torch.float32)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=d * d * (1 - beta2))
                if beta1 > 0:
                    # Garante que exp_avg existe e está no device correto
                    if "exp_avg" not in state:  # Deve ter sido inicializado antes, mas por segurança
                        state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(grad_fp32, alpha=d * (1 - beta1))
                del grad_fp32  # Libera memória
                # --- FIM ATUALIZAÇÃO EMA ---

                # Calcula denom no device correto
                denom = exp_avg_sq.sqrt().add_(d * group["eps"])  # Add inplace

                # Decoupled weight decay (fallback)
                if group["weight_decay"] != 0 and decouple:
                    if p.dtype == torch.bfloat16 and getattr(self, "stochastic_rounding", False):
                        temp = p.data.to(torch.float32) * (-group["weight_decay"] * dlr)
                        add_stochastic_(p.data, temp)
                        del temp
                    else:
                        p.data.add_(p.data, alpha=-group["weight_decay"] * dlr)

                # Atualização do parâmetro (fallback)
                if p.dtype == torch.bfloat16 and getattr(self, "stochastic_rounding", False):
                    if beta1 > 0:
                        # Acessa exp_avg que já está no device correto
                        addcdiv_stochastic_(p.data, state["exp_avg"], denom, value=-dlr)
                    else:
                        addcdiv_stochastic_(p.data, grad, denom, value=-dlr * d)  # Usa grad original
                else:
                    if beta1 > 0:
                        # Acessa exp_avg que já está no device correto
                        p.data.addcdiv_(state["exp_avg"], denom, value=-dlr)
                    else:
                        p.data.addcdiv_(grad, denom, value=-dlr * d)  # Usa grad original

                state["step"] += 1
                p.grad = None  # Limpa gradiente

        group["k"] = k + 1
    return loss


# --- step_parameter (MODIFICADO PARA REVERTER CPU OFFLOAD) ---
def step_parameter(self: Prodigy, p: torch.Tensor, param_group: dict, param_index_in_group: int):
    """Performs the Prodigy update step for a single parameter (NO CPU OFFLOAD)."""
    grad = p.grad.data if p.grad is not None else None  # Store grad if it exists
    state = self.state[p]
    group = param_group

    # --- Group Accumulator Initialization ---
    current_k = group["k"]
    init_marker = f"_initialized_for_step_{current_k}"

    if init_marker not in group:
        group[init_marker] = True
        old_marker = f"_initialized_for_step_{current_k-1}"
        if old_marker in group:
            del group[old_marker]
        group["_current_d_denom"] = 0.0
        beta3 = group["beta3"] if group["beta3"] is not None else math.sqrt(group["betas"][1])
        group["_current_d_numerator"] = group.get("d_numerator", 0.0) * beta3  # Usa get com default 0.0 por segurança

    if grad is None:
        if param_index_in_group == len(group["params"]) - 1:
            group["k"] = current_k + 1
            if init_marker in group:
                del group[init_marker]
            if "_current_d_denom" in group:
                del group["_current_d_denom"]
            if "_current_d_numerator" in group:
                del group["_current_d_numerator"]
        # p.grad = None # Já é None
        return

    # --- Proceed only if grad is not None ---
    if group["lr"] == 0.0:
        p.grad = None
        return

    # --- Parameter State Initialization (if needed) ---
    if "step" not in state:
        state["step"] = 0
        # >>>>>>>>> REVERTIDO: Inicializa no device de 'p' <<<<<<<<<
        state["s"] = torch.zeros_like(p.data.flatten()[:: group["slice_p"]]).detach()
        if p.any():
            state["p0"] = p.flatten()[:: group["slice_p"]].detach().clone()
        else:
            state["p0"] = torch.tensor(0, device=p.device, dtype=p.dtype)

        # >>>>>>>>> REVERTIDO: Cria estados EMA no device de 'p', mantendo float32 <<<<<<<<<
        if group["betas"][0] > 0:
            state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32, device=p.device)
        state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float32, device=p.device)
        # >>>>>>>>> FIM DA REVERSÃO <<<<<<<<<

    # --- Extract state and group parameters ---
    beta1, beta2 = group["betas"]
    d = group["d"]
    lr = group["lr"]
    use_bias_correction = group["use_bias_correction"]
    decouple = group["decouple"]
    slice_p = group["slice_p"]
    safeguard_warmup = group["safeguard_warmup"]
    d0 = group["d0"]
    eps = group["eps"]
    beta3 = group["beta3"] if group["beta3"] is not None else math.sqrt(beta2)

    # Apply weight decay (coupled variant)
    if group["weight_decay"] != 0 and not decouple:
        grad.add_(p.data, alpha=group["weight_decay"])

    bias_correction = ((1 - beta2 ** (current_k + 1)) ** 0.5) / (1 - beta1 ** (current_k + 1)) if use_bias_correction else 1.0
    dlr = d * lr * bias_correction

    # --- Update Adam EMA stats ---
    # >>>>>>>>> REVERTIDO: Cálculos no device de 'p' <<<<<<<<<
    exp_avg_sq = state["exp_avg_sq"]  # Já está no device de p
    # Converte gradiente para float32 no device original para precisão
    grad_fp32 = grad.to(dtype=torch.float32)
    exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=d * d * (1 - beta2))
    if beta1 > 0:
        exp_avg = state["exp_avg"]  # Já está no device de p
        exp_avg.mul_(beta1).add_(grad_fp32, alpha=d * (1 - beta1))
    # del grad_fp32 # Opcional liberar memória aqui
    # >>>>>>>>> FIM DA REVERSÃO <<<<<<<<<

    # --- Update Prodigy state 's' and accumulate d_numerator/d_denominator ---
    s = state["s"]
    p0 = state["p0"]
    # Garante que sliced_grad está no mesmo device de s e p0 (que é o de p)
    sliced_grad = grad.flatten()[::slice_p]

    if lr > 0.0:
        # >>>>>>>>> Cálculo do dot product no device de 'p' <<<<<<<<<
        dot_product = torch.dot(sliced_grad, p0 - p.data.flatten()[::slice_p])
        group["_current_d_numerator"] += (d / d0) * dlr * dot_product.item()

        # Update 's' state (operações no device de 's')
        if safeguard_warmup:
            s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
        else:
            s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))

        # Accumulate denominator part (operação no device de 's')
        group["_current_d_denom"] += s.abs().sum().item()

    # --- Parameter Update ---
    # >>>>>>>>> REVERTIDO: Cálculo do denom no device de 'p' <<<<<<<<<
    # exp_avg_sq já está no device correto (float32)
    denom = exp_avg_sq.sqrt().add_(d * eps)  # add inplace
    # >>>>>>>>> FIM DA REVERSÃO <<<<<<<<<

    # Apply weight decay (decoupled variant)
    if group["weight_decay"] != 0 and decouple:
        if p.dtype == torch.bfloat16 and getattr(self, "stochastic_rounding", False):
            temp_decay = p.data.to(torch.float32) * (-group["weight_decay"] * dlr)
            add_stochastic_(p.data, temp_decay)
            del temp_decay
        else:
            p.data.add_(p.data, alpha=-group["weight_decay"] * dlr)

    # Perform the parameter step update
    # >>>>>>>>> REVERTIDO: Usa estados EMA que já estão no device de 'p' <<<<<<<<<
    if p.dtype == torch.bfloat16 and getattr(self, "stochastic_rounding", False):
        if beta1 > 0:
            if "exp_avg" in state:
                # state['exp_avg'] já está no device correto
                addcdiv_stochastic_(p.data, state["exp_avg"], denom, value=-dlr)
        else:
            # grad já está no device correto
            addcdiv_stochastic_(p.data, grad, denom, value=-dlr * d)
    else:  # FP32 or no stochastic rounding
        if beta1 > 0:
            if "exp_avg" in state:
                # state['exp_avg'] já está no device correto
                p.data.addcdiv_(state["exp_avg"], denom, value=-dlr)
        else:
            # grad já está no device correto
            p.data.addcdiv_(grad, denom, value=-dlr * d)
    # >>>>>>>>> FIM DA REVERSÃO <<<<<<<<<

    # Increment internal step counter for the parameter
    state["step"] += 1

    # --- Group State Update (only after processing the last parameter) ---
    if param_index_in_group == len(group["params"]) - 1:
        if init_marker in group and group.get("_current_d_denom", 0.0) > 0.0 and lr > 0.0:
            d_coef = group["d_coef"]
            growth_rate = group["growth_rate"]
            d_hat = d_coef * group["_current_d_numerator"] / group["_current_d_denom"]
            d_max = group.get("d_max", d)  # Usa d_max anterior ou d atual se não existir

            new_d = d
            if d == group["d0"]:
                new_d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            new_d = min(d_max, new_d * growth_rate)

            group["d"] = new_d
            group["d_max"] = d_max
            group["d_hat"] = d_hat
            group["d_numerator"] = group["_current_d_numerator"]
            group["d_denom"] = group["_current_d_denom"]

        group["k"] = current_k + 1
        if init_marker in group:
            del group[init_marker]
        if "_current_d_denom" in group:
            del group["_current_d_denom"]
        if "_current_d_numerator" in group:
            del group["_current_d_numerator"]

    p.grad = None  # Release gradient memory


# --- Funções restantes (não modificadas) ---
def train_mode(self: Prodigy, mode=True):
    """Sets the optimizer training mode."""
    pass


def eval_mode(self: Prodigy):
    """Sets the optimizer eval mode."""
    pass


def patch_prodigy(optimizer: Prodigy, stochastic_rounding: bool):
    """Applies patches to the Prodigy optimizer instance."""
    setattr(optimizer, "stochastic_rounding", stochastic_rounding)
    # IMPORTANTE: Decida qual step usar. step_prodigy é vetorizado, step_parameter é por parâmetro.
    # Se você estava usando step_parameter por causa das modificações na CPU,
    # agora talvez possa voltar a usar step_prodigy se ele for mais eficiente.
    # Vou manter step_parameter por enquanto, assumindo que era essa a intenção.
    # optimizer.step = step_prodigy.__get__(
    #     optimizer, type(optimizer)
    # )  # <- VERIFIQUE: usar step_prodigy ou step_parameter? step_prodigy parece mais completo/vetorizado.
    optimizer.step_parameter = step_parameter.__get__(
        optimizer, type(optimizer)
    )  # <- Mantém o binding, mas talvez não seja chamado se `step` for `step_prodigy`

    optimizer.train = train_mode.__get__(optimizer, type(optimizer))
    optimizer.eval = eval_mode.__get__(optimizer, type(optimizer))
    optimizer.supports_fused_back_pass = lambda: True
    setattr(optimizer, "is_schedule_free", True)
