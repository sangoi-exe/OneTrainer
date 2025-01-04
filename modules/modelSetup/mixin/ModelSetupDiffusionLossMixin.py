from abc import ABCMeta
from collections.abc import Callable

from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.module.HPSv2ScoreModel import HPSv2ScoreModel
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.AlignPropLoss import AlignPropLoss
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.loss.masked_loss import masked_losses
from modules.util.loss.vb_loss import vb_losses
from torch.utils.tensorboard import SummaryWriter

from torch import Tensor
from modules.util.loss.dynamic_loss_strength import LossTracker, DynamicLossStrength

import torch
import torch.nn.functional as F


class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
	__coefficients: DiffusionScheduleCoefficients | None
	__alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None
	__sigmas: Tensor | None

	def __init__(self):
		super().__init__()
		self.__align_prop_loss_fn = None
		self.__coefficients = None
		self.__alphas_cumprod_fun = None
		self.__sigmas = None
		self.tensorboard = None
		self.progress = None
		self.loss_tracker = LossTracker(window_size=100, use_mad=False)
		# Dynamic loss weighting
		self.dynamic_loss_strengthing = DynamicLossStrength(
			use_ema=False,
			ema_decay=0.9,
			outlier_threshold=3.0,
			mae_start=0.6,
			mae_end=0.2,
			mse_start=0.2,
			mse_end=0.6,
			log_start=0.2,
			log_end=0.2,
		)

	def __align_prop_losses(
		self,
		batch: dict,
		data: dict,
		config: TrainConfig,
		train_device: torch.device,
	):
		if self.__align_prop_loss_fn is None:
			dtype = data["predicted"].dtype

			match config.align_prop_loss:
				case AlignPropLoss.HPS:
					self.__align_prop_loss_fn = HPSv2ScoreModel(dtype)
				case AlignPropLoss.AESTHETIC:
					self.__align_prop_loss_fn = AestheticScoreModel()

			self.__align_prop_loss_fn.to(device=train_device, dtype=dtype)
			self.__align_prop_loss_fn.requires_grad_(False)
			self.__align_prop_loss_fn.eval()

		losses = 0

		match config.align_prop_loss:
			case AlignPropLoss.HPS:
				with torch.autocast(
					device_type=train_device.type, dtype=data["predicted"].dtype
				):
					losses = self.__align_prop_loss_fn(
						data["predicted"], batch["prompt"], train_device
					)
			case AlignPropLoss.AESTHETIC:
				losses = self.__align_prop_loss_fn(data["predicted"])

		return losses * config.align_prop_weight

	def __log_cosh_loss(
			self,
			pred: torch.Tensor,
			target: torch.Tensor,
	):
		diff = pred - target
		loss = diff + torch.nn.functional.softplus(-2.0*diff) - torch.log(torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device))
		return loss

	def __masked_losses(
			self,
			batch: dict,
			data: dict,
			config: TrainConfig,
	):
		
		progress = self.progress
		losses = 0

		mse_loss = torch.tensor(0.0, device=data["predicted"].device)
		mae_loss = torch.tensor(0.0, device=data["predicted"].device)
		log_cosh_loss = torch.tensor(0.0, device=data["predicted"].device)

		# MSE/L2 Loss
		if config.mse_strength != 0:
			mse_loss = masked_losses(
				losses=F.mse_loss(
					data["predicted"].to(dtype=torch.float32),
					data["target"].to(dtype=torch.float32),
					reduction="none",
				),
				mask=batch["latent_mask"].to(dtype=torch.float32),
				unmasked_weight=config.unmasked_weight,
				normalize_masked_area_loss=config.normalize_masked_area_loss,
			).mean([1, 2, 3])

		# MAE/L1 Loss
		if config.mae_strength != 0:
			mae_loss = masked_losses(
				losses=F.l1_loss(
					data["predicted"].to(dtype=torch.float32),
					data["target"].to(dtype=torch.float32),
					reduction="none",
				),
				mask=batch["latent_mask"].to(dtype=torch.float32),
				unmasked_weight=config.unmasked_weight,
				normalize_masked_area_loss=config.normalize_masked_area_loss,
			).mean([1, 2, 3])

		# log-cosh Loss
		if config.log_cosh_strength != 0:
			log_cosh_loss = masked_losses(
				losses=self.__log_cosh_loss(
					data["predicted"].to(dtype=torch.float32),
					data["target"].to(dtype=torch.float32),
				),
				mask=batch["latent_mask"].to(dtype=torch.float32),
				unmasked_weight=config.unmasked_weight,
				normalize_masked_area_loss=config.normalize_masked_area_loss,
			).mean([1, 2, 3])
		
		# Update LossTracker
		self.loss_tracker.update(mse_loss, mae_loss, log_cosh_loss)

		# Compute z-scores
		mse_z, mae_z, log_cosh_z = self.loss_tracker.compute_z_scores(
			mse_loss, mae_loss, log_cosh_loss
		)

		# Ajusta pesos dinamicamente + scheduler de prioridades
		mse_weight, mae_weight, log_cosh_weight = (
			self.dynamic_loss_strengthing.adjust_weights(
				mse_z, mae_z, log_cosh_z, config, progress
			)
		)

		if self.tensorboard != None:
			self.tensorboard.add_scalar(
				"sangoi/7mse",
				mse_weight,
				progress.global_step,
			)
			self.tensorboard.add_scalar(
				"sangoi/8mae",
				mae_weight,
				progress.global_step,
			)
			self.tensorboard.add_scalar(
				"sangoi/9log_cosh",
				log_cosh_weight,
				progress.global_step,
			)

		losses = (
			mse_loss * mse_weight * config.mse_strength
			+ mae_loss * mae_weight * config.mae_strength
			+ log_cosh_loss * log_cosh_weight * config.log_cosh_strength
		)

		# VB loss
		if config.vb_loss_strength != 0 and 'predicted_var_values' in data and self.__coefficients is not None:
			losses += masked_losses(
				losses=vb_losses(
					coefficients=self.__coefficients,
					x_0=data['scaled_latent_image'].to(dtype=torch.float32),
					x_t=data['noisy_latent_image'].to(dtype=torch.float32),
					t=data['timestep'],
					predicted_eps=data['predicted'].to(dtype=torch.float32),
					predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
				),
				mask=batch['latent_mask'].to(dtype=torch.float32),
				unmasked_weight=config.unmasked_weight,
				normalize_masked_area_loss=config.normalize_masked_area_loss,
			).mean([1, 2, 3]) * config.vb_loss_strength

		return losses

	def __unmasked_losses(
			self,
			batch: dict,
			data: dict,
			config: TrainConfig,
	):
		
		progress = self.progress
		losses = 0

		mse_loss = torch.tensor(0.0, device=data["predicted"].device)
		mae_loss = torch.tensor(0.0, device=data["predicted"].device)
		log_cosh_loss = torch.tensor(0.0, device=data["predicted"].device)

		# MSE/L2 Loss
		if config.mse_strength != 0:
			mse_loss = F.mse_loss(
				data["predicted"].to(dtype=torch.float32),
				data["target"].to(dtype=torch.float32),
				reduction="none",
			).mean([1, 2, 3])

		# MAE/L1 Loss
		if config.mae_strength != 0:
			mae_loss = F.l1_loss(
				data["predicted"].to(dtype=torch.float32),
				data["target"].to(dtype=torch.float32),
				reduction="none",
			).mean([1, 2, 3])

		# log-cosh Loss
		if config.log_cosh_strength != 0:
			log_cosh_loss = self.__log_cosh_loss(
				data["predicted"].to(dtype=torch.float32),
				data["target"].to(dtype=torch.float32),
			).mean([1, 2, 3])
		
		# Update LossTracker
		self.loss_tracker.update(mse_loss, mae_loss, log_cosh_loss)

		# Compute z-scores
		mse_z, mae_z, log_cosh_z = self.loss_tracker.compute_z_scores(
			mse_loss, mae_loss, log_cosh_loss
		)

		# Ajusta pesos dinamicamente + scheduler de prioridades
		mse_weight, mae_weight, log_cosh_weight = (
			self.dynamic_loss_strengthing.adjust_weights(
				mse_z, mae_z, log_cosh_z, config, progress
			)
		)

		# Aplica os strengths do config DEPOIS de multiplicar pelos pesos
		losses = (
			mse_loss * mse_weight * config.mse_strength
			+ mae_loss * mae_weight * config.mae_strength
			+ log_cosh_loss * log_cosh_weight * config.log_cosh_strength
		)

		# VB loss
		if config.vb_loss_strength != 0 and "predicted_var_values" in data:
			vb_loss = (
				vb_losses(
					coefficients=self.__coefficients,
					x_0=data["scaled_latent_image"].to(dtype=torch.float32),
					x_t=data["noisy_latent_image"].to(dtype=torch.float32),
					t=data["timestep"],
					predicted_eps=data["predicted"].to(dtype=torch.float32),
					predicted_var_values=data["predicted_var_values"].to(
						dtype=torch.float32
					),
				).mean([1, 2, 3])
				* config.vb_loss_strength
			)
			losses += vb_loss
		
		if config.masked_training and config.normalize_masked_area_loss:
			clamped_mask = torch.clamp(batch["latent_mask"], config.unmasked_weight, 1)
			mask_mean = clamped_mask.mean(dim=(1, 2, 3))
			losses /= mask_mean

		return losses

	def __snr(self, timesteps: Tensor, device: torch.device):
		if self.__coefficients:
			all_snr = (
				self.__coefficients.sqrt_alphas_cumprod
				/ self.__coefficients.sqrt_one_minus_alphas_cumprod
			) ** 2
			all_snr.to(device)
			snr = all_snr[timesteps]
		else:
			alphas_cumprod = self.__alphas_cumprod_fun(timesteps, 1)
			snr = alphas_cumprod / (1.0 - alphas_cumprod)

		return snr

	def __min_snr_weight(
		self, timesteps: Tensor, gamma: float, v_prediction: bool, device: torch.device
	) -> Tensor:
		snr = self.__snr(timesteps, device)
		min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
		# Denominator of the snr_weight increased by 1 if v-prediction is being used.
		if v_prediction:
			snr += 1.0
		snr_weight = (min_snr_gamma / snr).to(device)
		return snr_weight

	def __debiased_estimation_weight(
		self, timesteps: Tensor, v_prediction: bool, device: torch.device
	) -> Tensor:
		snr = self.__snr(timesteps, device)
		weight = snr
		# The line below is a departure from the original paper.
		# This is to match the Kohya implementation, see: https://github.com/kohya-ss/sd-scripts/pull/889
		# In addition, it helps avoid numerical instability.
		torch.clip(weight, max=1.0e3, out=weight)
		if v_prediction:
			weight += 1.0
		torch.rsqrt(weight, out=weight)
		return weight

	def __p2_loss_weight(
		self,
		timesteps: Tensor,
		gamma: float,
		v_prediction: bool,
		device: torch.device,
	) -> Tensor:
		snr = self.__snr(timesteps, device)
		if v_prediction:
			snr += 1.0
		return (1.0 + snr) ** -gamma

	def __sigma_loss_weight(
		self,
		timesteps: Tensor,
		device: torch.device,
	) -> Tensor:
		return self.__sigmas[timesteps].to(device=device)

	def __sangoi_loss_weighting(
		self,
		timesteps: Tensor,
		predicted: Tensor,
		target: Tensor,
		device: torch.device,
		tensorboard: SummaryWriter,
		gamma: float,
	):
		"""
		Função Sangoi Loss Weighting com reescalonamento [0,1] -> [gamma, 1].
		Se combined_weight_raw > 1, fica 1. Se < 0 (em teoria não deveria), fica 0.
		"""

		self.tensorboard = tensorboard
		progress = self.progress

		# 1) Cálculo base do combined_weight (sem clamp de gamma nem de 1):
		snr = self.__snr(timesteps, device)
		epsilon = 1e-8

		mape = torch.abs((target - predicted) / (target + epsilon))
		mape = torch.clamp(mape, min=0, max=1).mean(dim=[1, 2, 3])

		snr_weight = torch.log(snr + 1)
		mape_reward = 1 - mape
		raw_reward = torch.exp(-mape_reward * snr_weight)
		# Ex: pode dar valores na casa de 0.08, 0.2, 1.1, etc.

		# 2) Clampar para [0, 1].
		#    - Se ficar acima de 1, virará 1 (sem redução de loss).
		#    - Se ficar abaixo de 0, virará 0 (caso improvável, mas é segurança).
		clamped_reward = torch.clamp(raw_reward, min=0.0, max=1.0)

		# 3) Reescalar [0,1] para [gamma,1].
		#    Se gamma=0.5, então 0 -> 0.5, 1 -> 1, 0.2 -> 0.6, etc.
		reward = gamma + (1.0 - gamma) * clamped_reward

		# Logging no TensorBoard
		tensorboard.add_scalar(
			"sangoi/1mape_reward", mape_reward.mean().item(), progress.global_step
		)
		tensorboard.add_scalar(
			"sangoi/2raw_reward",
			raw_reward.mean().item(),
			progress.global_step,
		)
		tensorboard.add_scalar(
			"sangoi/3clamped_reward",
			clamped_reward.mean().item(),
			progress.global_step,
		)
		tensorboard.add_scalar(
			"sangoi/4reward",
			reward.mean().item(),
			progress.global_step,
		)

		return reward

	def _diffusion_losses(
		self,
		batch: dict,
		data: dict,
		config: TrainConfig,
		progress: TrainProgress,
		tensorboard: SummaryWriter,
		train_device: torch.device,
		betas: Tensor | None = None,
		alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None = None,
	) -> Tensor:
		
		self.progress = progress
		
		loss_weight = batch["loss_weight"]

		batch_size_scale = (
			1
			if config.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION]
			else config.batch_size
		)
		gradient_accumulation_steps_scale = (
			1
			if config.loss_scaler in [LossScaler.NONE, LossScaler.BATCH]
			else config.gradient_accumulation_steps
		)

		if self.__coefficients is None and betas is not None:
			self.__coefficients = DiffusionScheduleCoefficients.from_betas(betas)

		self.__alphas_cumprod_fun = alphas_cumprod_fun

		if data["loss_type"] == "align_prop":
			losses = self.__align_prop_losses(batch, data, config, train_device)
		else:
			# TODO: don't disable masked loss functions when has_conditioning_image_input is true.
			#  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
			if (
				config.masked_training
				and not config.model_type.has_conditioning_image_input()
			):
				losses = self.__masked_losses(batch, data, config)
			else:
				losses = self.__unmasked_losses(batch, data, config)

		# Scale Losses by Batch and/or GA (if enabled)
		losses = losses * batch_size_scale * gradient_accumulation_steps_scale

		losses *= loss_weight.to(device=losses.device, dtype=losses.dtype)

		# Apply timestep based loss weighting.
		if "timestep" in data and data["loss_type"] != "align_prop":
			v_pred = data.get("prediction_type", "") == "v_prediction"
			match config.loss_weight_fn:
				case LossWeight.MIN_SNR_GAMMA:
					losses *= self.__min_snr_weight(
						data["timestep"],
						config.loss_weight_strength,
						v_pred,
						losses.device,
					)
				case LossWeight.DEBIASED_ESTIMATION:
					losses *= self.__debiased_estimation_weight(
						data["timestep"], v_pred, losses.device
					)
				case LossWeight.P2:
					losses *= self.__p2_loss_weight(
						data["timestep"],
						config.loss_weight_strength,
						v_pred,
						losses.device,
					)
				case LossWeight.SANGOI:
					tensorboard.add_scalar(
						"sangoi/5loss_b4_sangoi",
						losses.mean().item(),
						self.progress.global_step,
					)
					losses *= self.__sangoi_loss_weighting(
						data["timestep"],
						data["predicted"],
						data["target"],
						losses.device,
						tensorboard,
						config.loss_weight_strength,
					)
					tensorboard.add_scalar(
						"sangoi/6loss_after_sangoi",
						losses.mean().item(),
						self.progress.global_step,
					)

		return losses

	def _flow_matching_losses(
		self,
		batch: dict,
		data: dict,
		config: TrainConfig,
		train_device: torch.device,
		sigmas: Tensor | None = None,
	) -> Tensor:
		loss_weight = batch["loss_weight"]
		batch_size_scale = (
			1
			if config.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION]
			else config.batch_size
		)
		gradient_accumulation_steps_scale = (
			1
			if config.loss_scaler in [LossScaler.NONE, LossScaler.BATCH]
			else config.gradient_accumulation_steps
		)

		if self.__sigmas is None and sigmas is not None:
			num_timesteps = sigmas.shape[0]
			all_timesteps = torch.arange(
				start=1,
				end=num_timesteps + 1,
				step=1,
				dtype=torch.int32,
				device=sigmas.device,
			)
			self.__sigmas = all_timesteps / num_timesteps

		if data["loss_type"] == "align_prop":
			losses = self.__align_prop_losses(batch, data, config, train_device)
		else:
			# TODO: don't disable masked loss functions when has_conditioning_image_input is true.
			#  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
			if (
				config.masked_training
				and not config.model_type.has_conditioning_image_input()
			):
				losses = self.__masked_losses(batch, data, config)
			else:
				losses = self.__unmasked_losses(batch, data, config)

		# Scale Losses by Batch and/or GA (if enabled)
		losses = losses * batch_size_scale * gradient_accumulation_steps_scale

		losses *= loss_weight.to(device=losses.device, dtype=losses.dtype)

		# Apply timestep based loss weighting.
		if "timestep" in data and data["loss_type"] != "align_prop":
			match config.loss_weight_fn:
				case LossWeight.SIGMA:
					losses *= self.__sigma_loss_weight(data["timestep"], losses.device)

		return losses
