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
    Prodigy = None # ALTERAÇÃO INSERIDA

# ALTERAÇÃO INSERIDA: Lógica do step_prodigy baseada no código original de prodigyopt.prodigy.step
#                    com adição de suporte a stochastic rounding para BF16.
def step_prodigy(self: Prodigy, closure=None):
    """Performs a single optimization step.

    Arguments:
        closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
    """
    loss = None
    if closure is not None:
        loss = closure()

    # iterate over each parameter group independently
    for group_idx, group in enumerate(self.param_groups):

        if group['lr'] == 0.0:
            continue

        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        lr = group['lr'] # now each group has its own LR
        use_bias_correction = group['use_bias_correction']
        safeguard_warmup = group['safeguard_warmup']
        fsdp_in_use = group['fsdp_in_use']
        slice_p = group['slice_p']
        growth_rate = group['growth_rate']
        decouple = group['decouple']

        # group's iteration counter (k)
        k = group['k']

        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k+1)) ** 0.5) / (1 - beta1 ** (k+1))
        else:
            bias_correction = 1.0

        dlr = d * lr * bias_correction

        # we use the group's local d_numerator and locally reset the denominator
        d_numerator = group['d_numerator']
        d_numerator *= beta3

        d_denom = 0.0  # group's denominator (will accumulate terms)

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad.data

            # Apply weight decay (coupled variant)
            if group['weight_decay'] != 0 and not decouple:
                grad.add_(p.data, alpha=group['weight_decay'])

            # State initialization
            state = self.state[p]
            if 'step' not in state:
                state['step'] = 0
                state['s'] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()

                if p.any():
                    state['p0'] = p.flatten()[::slice_p].detach().clone()
                else:
                    # All values are zero, so save VRAM with a zero-tensor
                    state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

                # Exponential moving average of gradient values
                if beta1 > 0:
                    state['exp_avg'] = torch.zeros_like(p.data).detach()

                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data).detach()

            exp_avg_sq = state['exp_avg_sq']
            s = state['s']
            p0 = state['p0']

            # only if the group's LR is > 0 do we accumulate statistics
            if lr > 0.0:
                d0 = group['d0']
                # we use d / d0 instead of just d to avoid getting values that are too small
                sliced_grad = grad.flatten()[::slice_p]
                d_numerator += (d / d0) * dlr * torch.dot(sliced_grad, p0 - p.data.flatten()[::slice_p]).item()

                # Adam EMA updates
                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(grad, alpha=d * (1 - beta1))

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))

                if safeguard_warmup:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                else:
                    s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))

                # accumulate the total denominator for this group
                d_denom += s.abs().sum().item()

        # if we didn't accumulate any grad (e.g., grad = 0), skip this group
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if d_denom == 0.0:
            group['k'] = k + 1
            continue

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).to(next(p for p in group['params'] if p.grad is not None).device)
                dist_tensor[0] = d_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = dist_tensor[0]
                global_d_denom = dist_tensor[1]
            else:
                global_d_numerator = d_numerator
                global_d_denom = d_denom

            # compute d_hat and d_max for the current group
            d_hat = d_coef * global_d_numerator / global_d_denom

            # if it's the group's first update (d == d0), enforce d >= d_hat
            if d == group['d0']:
                d = max(d, d_hat)

            # also don't let d "fall" below d, but limit with growth_rate
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

            # store everything back in the group
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom

        # recompute dlr with the updated d (for this group)
        dlr = d * lr * bias_correction

        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
            state = self.state[p]
            exp_avg_sq = state['exp_avg_sq']

            state['step'] += 1

            denom = exp_avg_sq.sqrt().add_(d * group['eps'])

            # Apply weight decay (decoupled variant)
            if group['weight_decay'] != 0 and decouple:
                # Use add_stochastic_ for BF16 if enabled
                if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                     # Cannot use add_stochastic_ directly for weight decay as it modifies input
                     temp_decay = p.data.to(torch.float32) * (-group['weight_decay'] * dlr)
                     add_stochastic_(p.data, temp_decay)
                     del temp_decay
                else:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * dlr)


            ### Take step
            # Use addcdiv_stochastic_ for BF16 if enabled
            if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    addcdiv_stochastic_(p.data, exp_avg, denom, value=-dlr)
                else:
                    addcdiv_stochastic_(p.data, grad, denom, value=-dlr * d)
            else: # Original logic for FP32 or other types
                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr * d)


        # Increment the group's k
        group['k'] = k + 1

    return loss
# FIM DA ALTERAÇÃO INSERIDA

# ALTERAÇÃO INSERIDA: Implementação de step_parameter para fused backward pass
def step_parameter(self: Prodigy, p: torch.Tensor, param_group: dict, param_index_in_group: int):
    """Performs the Prodigy update step for a single parameter."""
    if p.grad is None:
        return

    grad = p.grad.data
    state = self.state[p]
    group = param_group

    # Early exit if LR is 0
    if group['lr'] == 0.0:
        p.grad = None # Release gradient even if no step is taken
        return

    # Ensure state is initialized
    if 'step' not in state:
        # Initialize state lazily if not done before
        state['step'] = 0
        state['s'] = torch.zeros_like(p.data.flatten()[::group['slice_p']]).detach()
        if p.any():
            state['p0'] = p.flatten()[::group['slice_p']].detach().clone()
        else:
            state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)
        if group['betas'][0] > 0:
            state['exp_avg'] = torch.zeros_like(p.data).detach()
        state['exp_avg_sq'] = torch.zeros_like(p.data).detach()
        # Initialize group denominator accumulator if this is the first param of the group being processed in this step
        if param_index_in_group == 0:
             group['_current_d_denom'] = 0.0
             group['_current_d_numerator'] = group['d_numerator'] * group['beta3'] if group.get('beta3') is not None else group['d_numerator'] * math.sqrt(group['betas'][1])


    # Extract state and group parameters
    beta1, beta2 = group['betas']
    beta3 = group['beta3'] if group['beta3'] is not None else math.sqrt(beta2)
    d = group['d']
    lr = group['lr']
    k = group['k']
    use_bias_correction = group['use_bias_correction']
    decouple = group['decouple']
    slice_p = group['slice_p']
    safeguard_warmup = group['safeguard_warmup']
    d0 = group['d0']
    eps = group['eps']

    # Apply weight decay (coupled variant) - Must happen before grad is used for EMA
    if group['weight_decay'] != 0 and not decouple:
        grad.add_(p.data, alpha=group['weight_decay'])

    # Bias correction factor
    bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1)) if use_bias_correction else 1.0
    dlr = d * lr * bias_correction # Use the 'd' from the *previous* step for the current update calculation

    # Update Adam EMA stats
    exp_avg_sq = state['exp_avg_sq']
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=d * d * (1 - beta2))

    if beta1 > 0:
        exp_avg = state['exp_avg']
        exp_avg.mul_(beta1).add_(grad, alpha=d * (1 - beta1))

    # Update Prodigy specific state 's' and accumulate denominator part
    s = state['s']
    p0 = state['p0']
    sliced_grad = grad.flatten()[::slice_p]

    if lr > 0.0:
        # Accumulate numerator part - uses d and dlr from *previous* step
        group['_current_d_numerator'] += (d / d0) * dlr * torch.dot(sliced_grad, p0 - p.data.flatten()[::slice_p]).item()

        # Update 's' state
        if safeguard_warmup:
            s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
        else:
            s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))

        # Accumulate denominator part
        group['_current_d_denom'] += s.abs().sum().item()


    # --- Parameter Update ---
    denom = exp_avg_sq.sqrt().add_(d * eps)

    # Apply weight decay (decoupled variant)
    if group['weight_decay'] != 0 and decouple:
         if p.dtype == torch.bfloat16 and self.stochastic_rounding:
             # Cannot use add_stochastic_ directly for weight decay as it modifies input
             temp_decay = p.data.to(torch.float32) * (-group['weight_decay'] * dlr) # Use dlr from previous step
             add_stochastic_(p.data, temp_decay)
             del temp_decay
         else:
             p.data.add_(p.data, alpha=-group['weight_decay'] * dlr) # Use dlr from previous step

    # Perform the parameter step update using dlr from *previous* step
    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
        if beta1 > 0:
            addcdiv_stochastic_(p.data, exp_avg, denom, value=-dlr)
        else:
            addcdiv_stochastic_(p.data, grad, denom, value=-dlr * d)
    else:
        if beta1 > 0:
            p.data.addcdiv_(exp_avg, denom, value=-dlr)
        else:
            p.data.addcdiv_(grad, denom, value=-dlr * d)

    # Increment internal step counter for the parameter
    state['step'] += 1

    # --- Group State Update (only after last parameter) ---
    if param_index_in_group == len(group['params']) - 1:
        if group['_current_d_denom'] > 0.0 and lr > 0.0: # Only update 'd' if grads were non-zero and lr > 0
             # Finalize 'd' calculation for the next step
             d_coef = group['d_coef']
             growth_rate = group['growth_rate']
             d_hat = d_coef * group['_current_d_numerator'] / group['_current_d_denom']
             d_max = group['d_max']

             # Update d, d_max for the *next* iteration
             new_d = d # Start with current d
             if d == group['d0']: # First update
                 new_d = max(d, d_hat)
             d_max = max(d_max, d_hat)
             new_d = min(d_max, new_d * growth_rate)

             # Store updated values back into the group state
             group['d'] = new_d
             group['d_max'] = d_max
             group['d_hat'] = d_hat # Store for potential logging/debugging
             group['d_numerator'] = group['_current_d_numerator'] # Store final numerator for next step's beta3 multiply
             group['d_denom'] = group['_current_d_denom'] # Store final denominator for logging/debugging

        # Increment the main group step counter 'k'
        group['k'] = k + 1
        # Clean up temporary accumulators
        del group['_current_d_denom']
        del group['_current_d_numerator']


    # Release gradient memory
    p.grad = None
# FIM DA ALTERAÇÃO INSERIDA

# ALTERAÇÃO INSERIDA: Funções dummy train/eval para compatibilidade com schedule-free
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
    from prodigyopt import Prodigy # Import locally to avoid circular dependency if used elsewhere

    optimizer.stochastic_rounding = stochastic_rounding
    optimizer.step = step_prodigy.__get__(optimizer, Prodigy)
    optimizer.step_parameter = step_parameter.__get__(optimizer, Prodigy)
    optimizer.train = train_mode.__get__(optimizer, Prodigy)
    optimizer.eval = eval_mode.__get__(optimizer, Prodigy)
    # Add supports_fused_back_pass method dynamically
    optimizer.supports_fused_back_pass = lambda: True
