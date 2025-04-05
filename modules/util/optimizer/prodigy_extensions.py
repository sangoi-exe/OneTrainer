# --- INÍCIO DO ARQUIVO modules/util/prodigy_extensions.py ---
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
    from prodigyopt import Prodigy
else:
    _params_t = Any
    # ALTERAÇÃO INSERIDA
    Prodigy = None

@torch.no_grad()
def step_prodigy(self: Prodigy, closure=None):
    loss = closure() if closure is not None else None

    for group in self.param_groups:
        if group['lr'] == 0.0:
            continue

        beta1, beta2 = group['betas']
        beta3 = group['beta3'] or math.sqrt(beta2)
        d, d_max, lr = group['d'], group['d_max'], group['lr']
        use_bias_correction = group['use_bias_correction']
        safeguard_warmup = group['safeguard_warmup']
        fsdp_in_use = group['fsdp_in_use']
        slice_p = group['slice_p']
        decouple = group['decouple']
        k = group['k']

        bias_correction = ((1 - beta2 ** (k+1)) ** 0.5) / (1 - beta1 ** (k+1)) if use_bias_correction else 1.0
        dlr = d * lr * bias_correction

        group['d_numerator'] *= beta3
        d_numerator = group['d_numerator']
        d_denom = 0.0

        # Vetorização: separa tensores válidos
        grads, params, exp_avgs, exp_avg_sqs, s_list, p0_list = [], [], [], [], [], []
        shapes_mismatch = False

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad.data
            state = self.state.setdefault(p, {})
            if 'step' not in state:
                state['step'] = 0
                state['s'] = torch.zeros_like(p.data.flatten()[::slice_p])
                state['p0'] = p.flatten()[::slice_p].clone() if p.any() else torch.tensor(0, device=p.device, dtype=p.dtype)
                if beta1 > 0:
                    state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

            # state refs
            s, p0 = state['s'], state['p0']
            if lr > 0.0:
                sliced_grad = grad.flatten()[::slice_p]
                d_numerator += (d / group['d0']) * dlr * torch.dot(sliced_grad, p0 - p.data.flatten()[::slice_p]).item()
                if safeguard_warmup:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / group['d0']) * d))
                else:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / group['d0']) * dlr))
                d_denom += s.abs().sum().item()

            # A partir daqui, preparamos vetorização se possível
            if p.dtype != torch.bfloat16 or not getattr(self, 'stochastic_rounding', False):
                grads.append(grad)
                params.append(p.data)
                exp_avg_sqs.append(self.state[p]['exp_avg_sq'])
                if beta1 > 0:
                    exp_avgs.append(self.state[p]['exp_avg'])
            else:
                shapes_mismatch = True  # fallback para laço manual

        if d_denom == 0.0:
            group['k'] = k + 1
            continue

        # Sincronização distribuída se necessário
        if fsdp_in_use:
            dist_tensor = torch.tensor([d_numerator, d_denom], device=params[0].device)
            dist.all_reduce(dist_tensor)
            d_numerator, d_denom = dist_tensor.tolist()

        d_hat = group['d_coef'] * d_numerator / d_denom
        if d == group['d0']:
            d = max(d, d_hat)
        d_max = max(d_max, d_hat)
        d = min(d_max, d * group['growth_rate'])

        # Salva novos valores
        group['d'] = d
        group['d_max'] = d_max
        group['d_hat'] = d_hat
        group['d_numerator'] = d_numerator
        group['d_denom'] = d_denom

        dlr = d * lr * bias_correction  # Recalcula após novo d

        # Atualizações vetorizadas se possível
        if grads and not shapes_mismatch:
            # Decoupled weight decay
            if group['weight_decay'] != 0 and decouple:
                torch._foreach_add_(params, params, alpha=-group['weight_decay'] * dlr)

            # EMA
            if beta1 > 0:
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads, alpha=d * (1 - beta1))
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=d * d * (1 - beta2))

            # denom = sqrt(v) + eps*d
            denom = [v.sqrt().add_(d * group['eps']) for v in exp_avg_sqs]
            if beta1 > 0:
                torch._foreach_addcdiv_(params, exp_avgs, denom, value=-dlr)
            else:
                torch._foreach_addcdiv_(params, grads, denom, value=-dlr * d)

            # Avança step
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None
                    self.state[p]['step'] += 1
        else:
            # Fallback (BF16 com stochastic_rounding)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg_sq = state['exp_avg_sq']
                denom = exp_avg_sq.sqrt().add_(d * group['eps'])

                if group['weight_decay'] != 0 and decouple:
                    if p.dtype == torch.bfloat16 and getattr(self, 'stochastic_rounding', False):
                        temp = p.data.to(torch.float32) * (-group['weight_decay'] * dlr)
                        add_stochastic_(p.data, temp)
                        del temp
                    else:
                        p.data.add_(p.data, alpha=-group['weight_decay'] * dlr)

                if p.dtype == torch.bfloat16 and getattr(self, 'stochastic_rounding', False):
                    if beta1 > 0:
                        addcdiv_stochastic_(p.data, state['exp_avg'], denom, value=-dlr)
                    else:
                        addcdiv_stochastic_(p.data, grad, denom, value=-dlr * d)
                else:
                    if beta1 > 0:
                        p.data.addcdiv_(state['exp_avg'], denom, value=-dlr)
                    else:
                        p.data.addcdiv_(grad, denom, value=-dlr * d)
                state['step'] += 1                
                p.grad = None

        group['k'] = k + 1
    return loss

def step_parameter(self: Prodigy, p: torch.Tensor, param_group: dict, param_index_in_group: int):
    """Performs the Prodigy update step for a single parameter."""
    grad = p.grad.data if p.grad is not None else None # Store grad if it exists
    state = self.state[p]
    group = param_group

    # --- Group Accumulator Initialization ---
    # Ensure initialization happens ONCE per group per optimizer step 'k'.
    # Use a marker tied to the group's step counter 'k'.
    current_k = group['k']
    init_marker = f'_initialized_for_step_{current_k}'

    if init_marker not in group:
        group[init_marker] = True # Mark as initialized for this k
        # Clean up markers from the previous step (optional, saves minor memory)
        old_marker = f'_initialized_for_step_{current_k-1}'
        if old_marker in group:
            del group[old_marker]

        # Initialize the temporary accumulators for this step
        group['_current_d_denom'] = 0.0
        beta3 = group['beta3'] if group['beta3'] is not None else math.sqrt(group['betas'][1])
        # Start numerator accumulation using the persistent d_numerator from the *previous* step (k-1)
        group['_current_d_numerator'] = group['d_numerator'] * beta3

    # If grad is None, we skip the update logic but still need to handle the group state logic if it's the last param
    if grad is None:
        # Check if this is the last parameter of the group
        if param_index_in_group == len(group['params']) - 1:
            # Finalize the step for the group even if this param had no grad
            # Increment the main group step counter 'k'
            group['k'] = current_k + 1
            # Clean up the initialization marker and temporary accumulators
            if init_marker in group:
                del group[init_marker]
            if '_current_d_denom' in group: del group['_current_d_denom']
            if '_current_d_numerator' in group: del group['_current_d_numerator']
        # Ensure gradient is None (it already is, but for consistency)
        p.grad = None
        return # Skip update for this parameter

    # --- Proceed only if grad is not None ---

    # Early exit if Learning Rate is 0
    if group['lr'] == 0.0:
        p.grad = None # Release gradient memory
        return

    # --- Parameter State Initialization (if needed) ---
    if 'step' not in state:
        state['step'] = 0
        state['s'] = torch.zeros_like(p.data.flatten()[::group['slice_p']]).detach()
        if p.any():
            state['p0'] = p.flatten()[::group['slice_p']].detach().clone()
        else:
            state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)
        if group['betas'][0] > 0:
            state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32, device='cpu')
        state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32, device='cpu')

    # --- Extract state and group parameters ---
    beta1, beta2 = group['betas']
    d = group['d'] # Use 'd' from the previous step (k) for this update
    lr = group['lr']
    # k = current_k (already defined)
    use_bias_correction = group['use_bias_correction']
    decouple = group['decouple']
    slice_p = group['slice_p']
    safeguard_warmup = group['safeguard_warmup']
    d0 = group['d0']
    eps = group['eps']
    beta3 = group['beta3'] if group['beta3'] is not None else math.sqrt(beta2) # Recalculate beta3 just in case

    # Apply weight decay (coupled variant) - Before grad is used for EMA
    if group['weight_decay'] != 0 and not decouple:
        grad.add_(p.data, alpha=group['weight_decay'])

    # Bias correction factor - Uses current_k
    bias_correction = ((1 - beta2 ** (current_k + 1)) ** 0.5) / (1 - beta1 ** (current_k + 1)) if use_bias_correction else 1.0
    # Calculate learning rate for *this* parameter update - based on d from step k
    dlr = d * lr * bias_correction

    # --- Update Adam EMA stats ---
    exp_avg_sq = state['exp_avg_sq']
    grad_cpu = grad.to('cpu', torch.float32)
    exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=d * d * (1 - beta2))
    if beta1 > 0:
        exp_avg = state['exp_avg']
        exp_avg.mul_(beta1).add_(grad_cpu, alpha=d * (1 - beta1))

    # --- Update Prodigy state 's' and accumulate d_numerator/d_denominator ---
    s = state['s']
    p0 = state['p0']
    sliced_grad = grad.flatten()[::slice_p]

    if lr > 0.0:
        # Accumulate numerator part - uses d and dlr from the previous step k
        # The key '_current_d_numerator' MUST exist now due to the initialization logic
        group['_current_d_numerator'] += (d / d0) * dlr * torch.dot(sliced_grad, p0 - p.data.flatten()[::slice_p]).item()

        # Update 's' state
        if safeguard_warmup:
            s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
        else:
            s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))

        # Accumulate denominator part - the key '_current_d_denom' MUST exist now
        group['_current_d_denom'] += s.abs().sum().item()

    # --- Parameter Update ---
    denom = exp_avg_sq.sqrt().add(d * eps).to(p.device)

    # Apply weight decay (decoupled variant) - using dlr from step k
    if group['weight_decay'] != 0 and decouple:
        # Check for stochastic rounding
        if p.dtype == torch.bfloat16 and getattr(self, 'stochastic_rounding', False):
            temp_decay = p.data.to(torch.float32) * (-group['weight_decay'] * dlr)
            add_stochastic_(p.data, temp_decay)
            del temp_decay
        else:
            p.data.add_(p.data, alpha=-group['weight_decay'] * dlr)

    # Perform the parameter step update - using dlr from step k
    # Check for stochastic rounding
    if p.dtype == torch.bfloat16 and getattr(self, 'stochastic_rounding', False):
        if beta1 > 0:
            # Ensure exp_avg exists before accessing
            if 'exp_avg' in state:
                addcdiv_stochastic_(p.data, state['exp_avg'].to(p.device), denom, value=-dlr)
            # else: Handle case where beta1 > 0 but exp_avg wasn't initialized? Should not happen with current logic.
        else:
            addcdiv_stochastic_(p.data, grad, denom, value=-dlr * d)
    else: # FP32 or no stochastic rounding
        if beta1 > 0:
             # Ensure exp_avg exists before accessing
            if 'exp_avg' in state:
                p.data.addcdiv_(state['exp_avg'].to(p.device), denom, value=-dlr)
            # else: Handle case?
        else:
            p.data.addcdiv_(grad, denom, value=-dlr * d)

    # Increment internal step counter for the parameter
    state['step'] += 1

    # --- Group State Update (only after processing the last parameter) ---
    if param_index_in_group == len(group['params']) - 1:
        # Check if the temporary accumulators exist (they should if init_marker was set)
        # Also check if denominator is non-zero and lr > 0 to perform the 'd' update
        if init_marker in group and group.get('_current_d_denom', 0.0) > 0.0 and lr > 0.0:
            # Finalize the calculation of 'd' for the *next* optimization step (k+1)
            d_coef = group['d_coef']
            growth_rate = group['growth_rate']
            # Use the temporary accumulators from *this* step (k)
            d_hat = d_coef * group['_current_d_numerator'] / group['_current_d_denom']
            d_max = group['d_max'] # d_max carries over from previous steps

            # Calculate the new 'd' value
            new_d = d # Start with the 'd' used in *this* step (from step k-1)
            if d == group['d0']: # Check if it was the first step for d
                new_d = max(d, d_hat)
            d_max = max(d_max, d_hat) # Update d_max
            new_d = min(d_max, new_d * growth_rate) # Apply growth rate limit

            # Store updated values back into the group's persistent state for step k+1
            group['d'] = new_d
            group['d_max'] = d_max
            group['d_hat'] = d_hat # Store d_hat for logging/debugging if needed
            # Store the final accumulated numerator in the persistent state for the *next* step's beta3 multiplication
            group['d_numerator'] = group['_current_d_numerator']
            group['d_denom'] = group['_current_d_denom'] # Store final denominator for logging

        # Increment the main group step counter 'k' regardless of whether 'd' was updated
        group['k'] = current_k + 1
        # Clean up the initialization marker and temporary accumulators
        if init_marker in group:
            del group[init_marker]
        if '_current_d_denom' in group: del group['_current_d_denom']
        if '_current_d_numerator' in group: del group['_current_d_numerator']

    # Release gradient memory for this parameter
    p.grad = None

def train_mode(self: Prodigy, mode=True):
    """Sets the optimizer training mode."""
    # Prodigy doesn't have distinct modes like schedulefree, but add method for API compatibility.
    pass

def eval_mode(self: Prodigy):
    """Sets the optimizer eval mode."""
    # Prodigy doesn't have distinct modes like schedulefree, but add method for API compatibility.
    pass

def patch_prodigy(optimizer: Prodigy, stochastic_rounding: bool):
    """Applies patches to the Prodigy optimizer instance."""
    # Import locally inside function if needed, or ensure it's imported at module level
    # from prodigyopt import Prodigy # Assuming Prodigy is the class from the library

    # Add or update the stochastic_rounding attribute
    setattr(optimizer, 'stochastic_rounding', stochastic_rounding)
    # Bind the custom step method
    optimizer.step = step_prodigy.__get__(optimizer, type(optimizer))
    # Bind the custom step_parameter method
    optimizer.step_parameter = step_parameter.__get__(optimizer, type(optimizer))
    # Bind dummy train/eval methods
    optimizer.train = train_mode.__get__(optimizer, type(optimizer))
    optimizer.eval = eval_mode.__get__(optimizer, type(optimizer))
    # Add supports_fused_back_pass method dynamically
    optimizer.supports_fused_back_pass = lambda: True
    # Mark as schedule-free (important for OneTrainer's logic)
    setattr(optimizer, 'is_schedule_free', True)