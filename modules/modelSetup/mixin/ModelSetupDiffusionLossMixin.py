from abc import ABCMeta
from collections.abc import Callable

from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.module.HPSv2ScoreModel import HPSv2ScoreModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.AlignPropLoss import AlignPropLoss
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.loss.masked_loss import masked_losses
from modules.util.loss.vb_loss import vb_losses

import torch
import torch.nn.functional as F
from torch import Tensor


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
                with torch.autocast(device_type=train_device.type, dtype=data["predicted"].dtype):
                    losses = self.__align_prop_loss_fn(data["predicted"], batch["prompt"], train_device)
            case AlignPropLoss.AESTHETIC:
                losses = self.__align_prop_loss_fn(data["predicted"])

        return losses * config.align_prop_weight

    def __log_cosh_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        diff = pred - target
        loss = (
            diff
            + torch.nn.functional.softplus(-2.0 * diff)
            - torch.log(torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device))
        )
        return loss

    def __masked_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ):
        losses = 0

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += (
                masked_losses(
                    losses=F.mse_loss(data["predicted"].to(dtype=torch.float32), data["target"].to(dtype=torch.float32), reduction="none"),
                    mask=batch["latent_mask"].to(dtype=torch.float32),
                    unmasked_weight=config.unmasked_weight,
                    normalize_masked_area_loss=config.normalize_masked_area_loss,
                ).mean([1, 2, 3])
                * config.mse_strength
            )

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += (
                masked_losses(
                    losses=F.l1_loss(data["predicted"].to(dtype=torch.float32), data["target"].to(dtype=torch.float32), reduction="none"),
                    mask=batch["latent_mask"].to(dtype=torch.float32),
                    unmasked_weight=config.unmasked_weight,
                    normalize_masked_area_loss=config.normalize_masked_area_loss,
                ).mean([1, 2, 3])
                * config.mae_strength
            )

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += (
                masked_losses(
                    losses=self.__log_cosh_loss(data["predicted"].to(dtype=torch.float32), data["target"].to(dtype=torch.float32)),
                    mask=batch["latent_mask"].to(dtype=torch.float32),
                    unmasked_weight=config.unmasked_weight,
                    normalize_masked_area_loss=config.normalize_masked_area_loss,
                ).mean([1, 2, 3])
                * config.log_cosh_strength
            )

        # VB loss
        if config.vb_loss_strength != 0 and "predicted_var_values" in data and self.__coefficients is not None:
            losses += (
                masked_losses(
                    losses=vb_losses(
                        coefficients=self.__coefficients,
                        x_0=data["scaled_latent_image"].to(dtype=torch.float32),
                        x_t=data["noisy_latent_image"].to(dtype=torch.float32),
                        t=data["timestep"],
                        predicted_eps=data["predicted"].to(dtype=torch.float32),
                        predicted_var_values=data["predicted_var_values"].to(dtype=torch.float32),
                    ),
                    mask=batch["latent_mask"].to(dtype=torch.float32),
                    unmasked_weight=config.unmasked_weight,
                    normalize_masked_area_loss=config.normalize_masked_area_loss,
                ).mean([1, 2, 3])
                * config.vb_loss_strength
            )

        return losses

    def __unmasked_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
    ):
        losses = 0

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += (
                F.mse_loss(data["predicted"].to(dtype=torch.float32), data["target"].to(dtype=torch.float32), reduction="none").mean(
                    [1, 2, 3]
                )
                * config.mse_strength
            )

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += (
                F.l1_loss(data["predicted"].to(dtype=torch.float32), data["target"].to(dtype=torch.float32), reduction="none").mean(
                    [1, 2, 3]
                )
                * config.mae_strength
            )

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += (
                self.__log_cosh_loss(data["predicted"].to(dtype=torch.float32), data["target"].to(dtype=torch.float32)).mean([1, 2, 3])
                * config.log_cosh_strength
            )

        # VB loss
        if config.vb_loss_strength != 0 and "predicted_var_values" in data:
            losses += (
                vb_losses(
                    coefficients=self.__coefficients,
                    x_0=data["scaled_latent_image"].to(dtype=torch.float32),
                    x_t=data["noisy_latent_image"].to(dtype=torch.float32),
                    t=data["timestep"],
                    predicted_eps=data["predicted"].to(dtype=torch.float32),
                    predicted_var_values=data["predicted_var_values"].to(dtype=torch.float32),
                ).mean([1, 2, 3])
                * config.vb_loss_strength
            )

        if config.masked_training and config.normalize_masked_area_loss:
            clamped_mask = torch.clamp(batch["latent_mask"], config.unmasked_weight, 1)
            mask_mean = clamped_mask.mean(dim=(1, 2, 3))
            losses /= mask_mean

        return losses

    def __snr(self, timesteps: Tensor, device: torch.device):
        if self.__coefficients:
            all_snr = (self.__coefficients.sqrt_alphas_cumprod / self.__coefficients.sqrt_one_minus_alphas_cumprod) ** 2
            all_snr.to(device)
            snr = all_snr[timesteps]
        else:
            alphas_cumprod = self.__alphas_cumprod_fun(timesteps, 1)
            snr = alphas_cumprod / (1.0 - alphas_cumprod)

        return snr

    def __min_snr_weight(self, timesteps: Tensor, gamma: float, v_prediction: bool, device: torch.device) -> Tensor:
        snr = self.__snr(timesteps, device)
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
        # Denominator of the snr_weight increased by 1 if v-prediction is being used.
        if v_prediction:
            snr += 1.0
        snr_weight = (min_snr_gamma / snr).to(device)
        return snr_weight

    """
    This is where the __min_snr_weight function was originally, but because I didn't use it, 
    I replaced it with my custom loss function, for laziness and convenience. But if you know 
    what you're doing, you can put it in any other place, or even modify the OT code enough 
    to add the function and activate it through the UI.
    """

    def __sangoi_loss_modifier(self, timesteps: Tensor, predicted: Tensor, target: Tensor, gamma: float, device: torch.device) -> Tensor:
        """
        Source: https://github.com/sangoi-exe/sangoi-loss-function

        Computes a loss modifier based on the Mean Absolute Percentage Error (MAPE) and the Signal-to-Noise Ratio (SNR).
        This modifier adjusts the loss according to the prediction accuracy and the difficulty of the prediction task.

        Args:
            timesteps (Tensor): The current training step's timesteps.
            predicted (Tensor): Predicted values from the neural network.
            target (Tensor): Ground truth target values.
            gamma (float): A scaling factor (unused in this function).
            device (torch.device): The device on which tensors are allocated.

        Returns:
            Tensor: A tensor of weights per example to modify the loss.
        """

        # Define minimum and maximum SNR values to clamp extreme values
        # min_snr = 1e-4
        # max_snr = 100

        # Obtain the SNR for each timestep
        snr = self.__snr(timesteps, device)
        # Clamp the SNR values to the defined range to avoid extreme values
        # snr = torch.clamp(snr, min=min_snr, max=max_snr)

        # Define a small epsilon to prevent division by zero
        epsilon = 1e-8
        # Compute the Mean Absolute Percentage Error (MAPE)
        mape = torch.abs((target - predicted) / (target + epsilon))
        # Normalize MAPE values between 0 and 1
        mape = torch.clamp(mape, min=0, max=1)
        # Calculate the average MAPE per example across spatial dimensions
        mape = mape.mean(dim=[1, 2, 3])

        # Compute the SNR weight using the natural logarithm (adding 1 to avoid log(0))
        snr_weight = torch.log(snr + 1)
        # Invert MAPE to represent accuracy instead of error
        mape_reward = 1 - mape
        # Calculate the combined weight using the negative exponential of the product of MAPE reward and SNR weight
        combined_weight = torch.exp(-mape_reward * snr_weight)

        # Return the tensor of weights per example to modify the loss
        return combined_weight

    def __debiased_estimation_weight(self, timesteps: Tensor, v_prediction: bool, device: torch.device) -> Tensor:
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

    def _diffusion_losses(
        self,
        batch: dict,
        data: dict,
        config: TrainConfig,
        train_device: torch.device,
        betas: Tensor | None = None,
        alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None = None,
    ) -> Tensor:
        loss_weight = batch["loss_weight"]
        batch_size_scale = 1 if config.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION] else config.batch_size
        gradient_accumulation_steps_scale = (
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.BATCH] else config.gradient_accumulation_steps
        )

        if self.__coefficients is None and betas is not None:
            self.__coefficients = DiffusionScheduleCoefficients.from_betas(betas)

        self.__alphas_cumprod_fun = alphas_cumprod_fun

        if data["loss_type"] == "align_prop":
            losses = self.__align_prop_losses(batch, data, config, train_device)
        else:
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
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
                    losses *= self.__sangoi_loss_modifier(
                        data["timestep"], data["predicted"], data["target"], config.loss_weight_strength, losses.device
                    )
                case LossWeight.DEBIASED_ESTIMATION:
                    losses *= self.__debiased_estimation_weight(data["timestep"], v_pred, losses.device)
                case LossWeight.P2:
                    losses *= self.__p2_loss_weight(data["timestep"], config.loss_weight_strength, v_pred, losses.device)

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
        batch_size_scale = 1 if config.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION] else config.batch_size
        gradient_accumulation_steps_scale = (
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.BATCH] else config.gradient_accumulation_steps
        )

        if self.__sigmas is None and sigmas is not None:
            num_timesteps = sigmas.shape[0]
            all_timesteps = torch.arange(start=1, end=num_timesteps + 1, step=1, dtype=torch.int32, device=sigmas.device)
            self.__sigmas = all_timesteps / num_timesteps

        if data["loss_type"] == "align_prop":
            losses = self.__align_prop_losses(batch, data, config, train_device)
        else:
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
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
