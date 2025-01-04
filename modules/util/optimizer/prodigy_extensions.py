import math
import torch
import torch.distributed as dist

from modules.util.bf16_stochastic_rounding import (
	add_stochastic_,
	addcdiv_stochastic_,
)

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
					p.data.add_(p.data, alpha=-group['weight_decay'] * dlr)

				### Take step
				if beta1 > 0:
					exp_avg = state['exp_avg']
					if p.dtype == torch.bfloat16 and self.stochastic_rounding:
						addcdiv_stochastic_(p.data, exp_avg, denom, value=-dlr)
					else:
						p.data.addcdiv_(exp_avg, denom, value=-dlr)
				else:
					if p.dtype == torch.bfloat16 and self.stochastic_rounding:
						addcdiv_stochastic_(p.data, grad, denom, value=-dlr * d)
					else:
						p.data.addcdiv_(grad, denom, value=-dlr * d)

			# Increment the group's k
			group['k'] = k + 1

		return loss

def patch_prodigy(optimizer: Prodigy, stochastic_rounding: bool):
	"""
	Ativa o suporte a Stochastic Rounding no Prodigy,
	substituindo o método step pelo nosso step_prodigy.
	"""
	optimizer.stochastic_rounding = stochastic_rounding
	optimizer.step = step_prodigy.__get__(optimizer, Prodigy)