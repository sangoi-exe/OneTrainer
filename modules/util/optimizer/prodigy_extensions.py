import math
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import torch.optim
import logging
import os
import torch.distributed as dist
from prodigyopt import Prodigy

from modules.util.bf16_stochastic_rounding import addcdiv_stochastic_, add_stochastic_

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prodigyopt import Prodigy  # só pra tipagem
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


@torch.no_grad()
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
            state['exp_avg'] = torch.zeros_like(p.data, dtype=p.dtype, device=p.device)
        state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=p.dtype, device=p.device)

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
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))
    if beta1 > 0:
        exp_avg = state['exp_avg']
        exp_avg.mul_(beta1).add_(grad, alpha=d * (1 - beta1))

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


def patch_prodigy(optimizer: Prodigy, stochastic_rounding: bool):
    setattr(optimizer, "stochastic_rounding", stochastic_rounding)

    # Conecta step_parameter
    optimizer.step_parameter = step_parameter.__get__(optimizer, type(optimizer))

    # Agora pode declarar como fused
    optimizer.supports_fused_back_pass = lambda: True
    optimizer.is_schedule_free = False
