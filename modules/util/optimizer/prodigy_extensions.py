import math
import torch
import torch.distributed as dist

# Importe as funções de arredondamento estocástico
from modules.util.bf16_stochastic_rounding import (
    add_stochastic_,
    addcdiv_stochastic_,
)

# Importe a classe Prodigy para a tipagem e para a substituição do método.
# (Ajuste o path conforme necessário, se for "from prodigyopt.prodigy import Prodigy" ou similar)
from prodigyopt.prodigy import Prodigy


@torch.no_grad()
def step_prodigy(self, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    # Itera sobre cada grupo de parâmetros *independentemente*
    for group_idx, group in enumerate(self.param_groups):

        # Se a learning rate desse grupo for zero, simplesmente não fazemos nada
        # e passamos para o próximo grupo
        if group["lr"] == 0.0:
            continue

        beta1, beta2 = group["betas"]
        beta3 = group["beta3"]
        if beta3 is None:
            beta3 = math.sqrt(beta2)

        d = group["d"]
        d_max = group["d_max"]
        d_coef = group["d_coef"]
        lr = group["lr"]  # Agora cada grupo possui sua própria LR
        use_bias_correction = group["use_bias_correction"]
        safeguard_warmup = group["safeguard_warmup"]
        fsdp_in_use = group["fsdp_in_use"]
        slice_p = group["slice_p"]
        growth_rate = group["growth_rate"]
        decouple = group["decouple"]

        # Contador de iterações do grupo (k)
        k = group["k"]

        # Fator de bias correction do Adam (opcional)
        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1.0

        # Ajuste de escala final (equivalente a d * lr * bias_correction)
        dlr = d * lr * bias_correction

        # Usamos d_numerator local do grupo e zera localmente o denominador
        d_numerator = group["d_numerator"]
        d_numerator *= beta3  # Decaimento exponencial

        d_denom = 0.0  # Denominador do grupo (vai acumular termos)

        # -----------------------------
        # 1) Loop para coletar estatísticas e acumular d_numerator/d_denom
        # -----------------------------
        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.data

            # Aplicar weight decay acoplado (como no Adam clássico), se aplicável
            if group["weight_decay"] != 0 and not decouple:
                grad.add_(p.data, alpha=group["weight_decay"])

            # Inicializa o state do parâmetro, se ainda não existir
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["s"] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()

                if p.any():
                    state["p0"] = p.flatten()[::slice_p].detach().clone()
                else:
                    # Se todos os valores são zero, não precisa guardar grande tensor
                    state["p0"] = torch.tensor(0, device=p.device, dtype=p.dtype)

                if beta1 > 0:
                    state["exp_avg"] = torch.zeros_like(p.data).detach()
                state["exp_avg_sq"] = torch.zeros_like(p.data).detach()

            exp_avg_sq = state["exp_avg_sq"]
            s = state["s"]
            p0 = state["p0"]

            # Somente se a LR do grupo for > 0 vamos acumular estatísticas
            if lr > 0.0:
                # d / d0 para “normalizar” D na fase inicial
                d0 = group["d0"]
                sliced_grad = grad.flatten()[::slice_p]
                d_numerator += (d / d0) * dlr * torch.dot(sliced_grad, p0 - p.data.flatten()[::slice_p]).item()

                # EMA do grad (beta1)
                if beta1 > 0:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(beta1).add_(grad, alpha=d * (1 - beta1))

                # EMA do grad^2 (beta2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))

                # Atualiza s (usado no denominador do D)
                if safeguard_warmup:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                else:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))

                # Acumula o denominador total para este grupo
                d_denom += s.abs().sum().item()

        # Se não acumulamos nada de grad (ex: grad = 0), pula esse grupo
        if d_denom == 0.0:
            # Atualiza contagem de passos e segue
            group["k"] = k + 1
            continue

        # -----------------------------
        # 2) Ajuste de d (D-adaptation) para este grupo
        # -----------------------------
        if lr > 0.0:
            if fsdp_in_use:
                # Caso estejamos usando FSDP e precisemos reduzir de todos os processos
                dist_tensor = torch.zeros(2).to(next(p for p in group["params"] if p.grad is not None).device)
                dist_tensor[0] = d_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator
                global_d_denom = d_denom

            # Faz o cálculo do d_hat e do d_max do grupo atual
            d_hat = d_coef * global_d_numerator / global_d_denom

            # Se for a primeira atualização do grupo (d == d0), impõe d >= d_hat
            if d == group["d0"]:
                d = max(d, d_hat)

            # Também não deixa o d “cair” abaixo de d, mas limita com growth_rate
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

            # Armazena tudo de volta no grupo
            group["d"] = d
            group["d_max"] = d_max
            group["d_hat"] = d_hat
            group["d_numerator"] = global_d_numerator
            group["d_denom"] = global_d_denom

        # Recalcula dlr com o d atualizado (para este grupo)
        dlr = d * lr * bias_correction

        # -----------------------------
        # 3) Aplica a atualização final aos parâmetros do grupo
        # -----------------------------
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[p]
            exp_avg_sq = state["exp_avg_sq"]

            state["step"] += 1

            # denom = sqrt(EMA(grad^2)) + d*eps
            denom = exp_avg_sq.sqrt().add_(d * group["eps"])

            # Weight decay (decoupled)
            if group["weight_decay"] != 0 and decouple:
                if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                    add_stochastic_(p.data, p.data, alpha=-group["weight_decay"] * dlr)
                else:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * dlr)

            # Aqui a única vez que aplicamos addcdiv_ ou addcdiv_stochastic_
            if beta1 > 0:
                # Usa exp_avg
                exp_avg = state['exp_avg']
                if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                    addcdiv_stochastic_(p.data, exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
            else:
                # Usa grad, multiplicado por d
                if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                    addcdiv_stochastic_(p.data, grad, denom, value=-dlr * d)
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr * d)

        # Incrementa o k do grupo
        group["k"] = k + 1

    return loss


def patch_prodigy(optimizer: Prodigy, stochastic_rounding: bool):
    """
    Ativa o suporte a Stochastic Rounding no Prodigy,
    substituindo o método step pelo nosso step_prodigy.
    """
    optimizer.stochastic_rounding = stochastic_rounding
    optimizer.step = step_prodigy.__get__(optimizer, Prodigy)
