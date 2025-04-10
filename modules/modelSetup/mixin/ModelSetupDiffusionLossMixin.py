from abc import ABCMeta
from collections.abc import Callable
import os

from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.module.HPSv2ScoreModel import HPSv2ScoreModel
from modules.util.TrainProgress import TrainProgress
from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.loss.masked_loss import masked_losses
from modules.util.loss.vb_loss import vb_losses
from torch.utils.tensorboard import SummaryWriter

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter # Adicionado para type hint
from modules.util.TrainProgress import TrainProgress # Adicionado para type hint
from modules.util.config.TrainConfig import TrainConfig # Adicionado para type hint
from modules.util.loss.dynamic_loss_strength import LossTracker, DynamicLossStrength, DeltaPatternRegularizer
from modules.util.NamedParameterGroup import NamedParameterGroupCollection

import torch
import torch.nn.functional as F


class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
  __coefficients: DiffusionScheduleCoefficients | None
  __alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None
  __sigmas: Tensor | None
  config: TrainConfig | None # Adicionado tipo para clareza
  progress: TrainProgress | None # Adicionado tipo para clareza
  tensorboard: SummaryWriter | None # Adicionado tipo para clareza
  delta_pattern: DeltaPatternRegularizer | None # Movido para __init__

  def __init__(self):
    super().__init__()
    self.__coefficients = None
    self.__alphas_cumprod_fun = None
    self.__sigmas = None
    self.tensorboard = None
    self.progress = None
    self.config = None
    self.delta_pattern = None
    self.loaded_pattern_deltas = None
    self.loss_tracker = LossTracker(window_size=100, use_mad=False)
    self.dynamic_loss_strengthing = DynamicLossStrength()

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
    if config.mse_strength != 0 or config.loss_mode_fn == "SANGOI":
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
    if config.mae_strength != 0 or config.loss_mode_fn == "SANGOI":
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
    if config.log_cosh_strength != 0 or config.loss_mode_fn == "SANGOI":
      log_cosh_loss = masked_losses(
        losses=self.__log_cosh_loss(
          data["predicted"].to(dtype=torch.float32),
          data["target"].to(dtype=torch.float32),
        ),
        mask=batch["latent_mask"].to(dtype=torch.float32),
        unmasked_weight=config.unmasked_weight,
        normalize_masked_area_loss=config.normalize_masked_area_loss,
      ).mean([1, 2, 3])

    match config.loss_mode_fn:
      case config.loss_mode_fn.ORIGINAL:
        losses = (
          mse_loss * config.mse_strength +
          mae_loss * config.mae_strength +
          log_cosh_loss * config.log_cosh_strength
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

      case config.loss_mode_fn.SANGOI:
        # Update LossTracker
        self.loss_tracker.update(mse_loss, mae_loss, log_cosh_loss)

        # Compute z-scores
        mse_z, mae_z, log_cosh_z = self.loss_tracker.compute_z_scores(mse_loss, mae_loss, log_cosh_loss)

        # Ajusta pesos dinamicamente + scheduler de prioridades
        mse_weight, mae_weight, log_cosh_weight = self.dynamic_loss_strengthing.adjust_weights(
          mse_z, mae_z, log_cosh_z, config, progress
        )
        losses = (
          mse_loss * mse_weight * config.mse_strength
          + mae_loss * mae_weight * config.mae_strength
          + log_cosh_loss * log_cosh_weight * config.log_cosh_strength
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
    if config.mse_strength != 0 or config.loss_mode_fn == "SANGOI":
      mse_loss = F.mse_loss(
        data["predicted"].to(dtype=torch.float32),
        data["target"].to(dtype=torch.float32),
        reduction="none",
      ).mean([1, 2, 3])

    # MAE/L1 Loss
    if config.mae_strength != 0 or config.loss_mode_fn == "SANGOI":
      mae_loss = F.l1_loss(
        data["predicted"].to(dtype=torch.float32),
        data["target"].to(dtype=torch.float32),
        reduction="none",
      ).mean([1, 2, 3])

    # log-cosh Loss
    if config.log_cosh_strength != 0 or config.loss_mode_fn == "SANGOI":
      log_cosh_loss = self.__log_cosh_loss(
        data["predicted"].to(dtype=torch.float32),
        data["target"].to(dtype=torch.float32),
      ).mean([1, 2, 3])
    
    match config.loss_mode_fn:
      case config.loss_mode_fn.ORIGINAL:
        losses = (
          mse_loss * config.mse_strength +
          mae_loss * config.mae_strength +
          log_cosh_loss * config.log_cosh_strength
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

      case config.loss_mode_fn.SANGOI:
        # Update LossTracker
        self.loss_tracker.update(mse_loss, mae_loss, log_cosh_loss)

        # Compute z-scores
        mse_z, mae_z, log_cosh_z = self.loss_tracker.compute_z_scores(mse_loss, mae_loss, log_cosh_loss)

        # Ajusta pesos dinamicamente + scheduler de prioridades
        mse_weight, mae_weight, log_cosh_weight = self.dynamic_loss_strengthing.adjust_weights(
          mse_z, mae_z, log_cosh_z, config, progress
        )
        losses = (
          mse_loss * mse_weight * config.mse_strength
          + mae_loss * mae_weight * config.mae_strength
          + log_cosh_loss * log_cosh_weight * config.log_cosh_strength
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
    config = self.config

    # 1) Cálculo do snr "padrão"
    snr = self.__snr(timesteps, device)        # (batch, ...)
    epsilon = 1e-8

    # 2) Cálculo do MAPE (já presente)
    # 2) Blend MAPE + MSPE (peso 50/50)
    abs_percent_error = torch.abs((target - predicted) / (target + epsilon)).clamp(min=0, max=1)
    sq_percent_error = abs_percent_error ** 2
    blended_error = 0.5 * abs_percent_error + 0.5 * sq_percent_error
    mape = blended_error.mean(dim=[1, 2, 3])

    # -----------------------------------------------------------------------
    # CÁLCULO DO FATOR DE PROGRESSO
    # -----------------------------------------------------------------------
    # Se total_epochs = N, progress.epoch vai de 0 até N-1 (ou 1 até N, depende do trainer).
    # Ajuste conforme o comportamento real do seu `progress.epoch`.
    total_epochs = config.epochs
    current_epoch = progress.epoch  # verifique se vai de 0 a N-1 ou 1 a N
    # Fazemos um clamp para evitar divisões por zero em caso de 1 época só:
    if total_epochs <= 1:
      alpha = 1.0
    else:
      alpha = current_epoch / float(total_epochs - 1)  # varia de 0 até 1

    # -----------------------------------------------------------------------
    # CRIANDO DOIS "EXTREMOS" DE PESO PARA O SNR
    # -----------------------------------------------------------------------
    # A ideia é que no início do treino (alpha ~ 0),
    # queremos enfatizar cenários de SNR baixo como "mais difíceis".
    # No fim do treino (alpha ~ 1), enfatizamos cenários de SNR alto como "mais difíceis".
    #
    # Aqui vai um exemplo de forma de interpolar:
    # snr_weight_low_first  => enfatiza SNR BAIXO como "mais difícil"
    # snr_weight_high_first => enfatiza SNR ALTO  como "mais difícil"
    #
    # Você pode escolher a fórmula que fizer mais sentido para o seu caso.
    #
    # Exemplo de uma forma simples:
    #  - Se snr estiver alto, snr_weight_low_first deve ser PEQUENO.
    #  - Se snr estiver baixo, snr_weight_low_first deve ser MAIOR.
    #
    # Uma abordagem é usar: snr_weight_low_first = log(1 + 1/(snr+eps)),
    # pois, para snr grande, 1/(snr+eps) ≈ 0, resultando em log(1)≈0 (cenário "fácil").
    # E para snr pequeno, 1/(snr+eps) é grande, resultando em log(...) maior (cenário "difícil").
    #
    # Por outro lado, snr_weight_high_first = log(1 + snr)
    # faz o contrário: para snr grande, o log é grande; para snr pequeno, o log é pequeno.
    #
    # Depois, interpolamos linearmente entre esses dois extremos pelo fator alpha.

    snr_weight_low_first = torch.log(1.0 + 1.0 / (snr + epsilon))  # enfatiza SNR baixo
    snr_weight_high_first = torch.log(snr + 1.0)                   # enfatiza SNR alto

    # Interpolação linear:
    # alpha=0 => weight = snr_weight_low_first
    # alpha=1 => weight = snr_weight_high_first
    scenario_snr_weight = (1.0 - alpha) * snr_weight_low_first + alpha * snr_weight_high_first
    mape_reward = 1 - mape
    raw_reward = torch.exp(-mape_reward * scenario_snr_weight)
    # Ex: pode dar valores na casa de 0.08, 0.2, 1.1, etc.

    # 2) Clampar para [0, 1]
    clamped_reward = torch.clamp(raw_reward, min=0.0, max=1.0)

    # 3) Reescalar [0,1] para [gamma,1]
    reward = gamma + (1.0 - gamma) * clamped_reward

    # Logging no TensorBoard
    tensorboard.add_scalar(
      "sangoi/1mape_reward", mape_reward.mean().item(), progress.global_step
    )
    tensorboard.add_scalar(
      "sangoi/2scenario_snr_weight", scenario_snr_weight.mean().item(), progress.global_step
    )
    tensorboard.add_scalar(
      "sangoi/3clamped_reward", clamped_reward.mean().item(), progress.global_step
    )
    tensorboard.add_scalar(
      "sangoi/4reward", reward.mean().item(), progress.global_step
    )
    tensorboard.add_scalar(
      "sangoi/alpha", alpha, progress.global_step
    )
    tensorboard.add_scalar(
      "sangoi/scenario_snr_weight_mean",
      scenario_snr_weight.mean().item(),
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
    model: torch.nn.Module,
    betas: Tensor | None = None,
    alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None = None,
  ) -> Tensor:
    
    self.config = config
    self.progress = progress
    self.tensorboard = tensorboard

    if self.delta_pattern is None and (config.delta_pattern_save_it or config.delta_pattern_use_it):
      # --- PONTO CRÍTICO: Obter a coleção de parâmetros ---
      # Tenta acessar um atributo comum; ajuste se o nome for diferente em OneTrainer.
      param_collection = getattr(model, 'parameter_groups', None)
      if not isinstance(param_collection, NamedParameterGroupCollection):
        # Tenta outro nome comum ou gera um erro/aviso
        param_collection = getattr(model, 'param_groups', None) # Exemplo alternativo
        if not isinstance(param_collection, NamedParameterGroupCollection):
          raise AttributeError("Could not find NamedParameterGroupCollection on the model. Adjust attribute name (e.g., 'parameter_groups') in lossMixin.py")
      print("[DeltaPattern] Initializing DeltaPatternRegularizer...")
      # Passa a coleção de parâmetros para o regularizador
      self.delta_pattern = DeltaPatternRegularizer(param_collection)
      # O resto da lógica de captura/carregamento permanece a mesma
      if config.delta_pattern_save_it:
        print("[DeltaPattern] Capturing initial weights for delta calculation (Run 1).")
        self.delta_pattern.capture_weights()
      if config.delta_pattern_use_it:
        if config.delta_pattern_path and os.path.exists(config.delta_pattern_path):
          print(f"[DeltaPattern] Loading reference delta pattern from: {config.delta_pattern_path}")
          self.delta_pattern.load_reference_pattern(config.delta_pattern_path)
          print("[DeltaPattern] Capturing initial weights for penalty calculation (Run 2).")
          self.delta_pattern.capture_initial_weights_run2()
        else:
          print(f"[DeltaPattern] Warning: 'delta_pattern_use_it' is True, but path '{config.delta_pattern_path}' not found or not specified. Penalty will be disabled.")
          config.delta_pattern_use_it = False
    
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
      # losses = self.__align_prop_losses(batch, data, config, train_device) # Função não fornecida, mantendo placeholder
      raise NotImplementedError("AlignProp foi removido e eu to com preguiça de ajeitar esse if-else bosta.") # Adicionado para clareza
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

      # INÍCIO ALTERAÇÃO: Aplicação da penalidade Delta Pattern
      # Aplicado *depois* do loss weighting normal, mas antes do retorno
      if config.delta_pattern_use_it and self.delta_pattern is not None and self.delta_pattern.reference_deltas:
        # Calcula a penalidade usando os pesos *atuais* do modelo
        # e comparando o delta *acumulado* com o delta de referência
        penalty = self.delta_pattern.compute_penalty(
          lambda_weight=config.delta_pattern_weight
        )
        # Adiciona a penalidade à loss média do batch
        # Precisamos garantir que a penalidade tenha a mesma dimensão ou seja broadcastable
        # Assumindo que 'losses' é (batch_size), a penalidade é um escalar
        losses += penalty # Adiciona o escalar à loss de cada item do batch

        if self.tensorboard:
          self.tensorboard.add_scalar(
            "delta_pattern/penalty", penalty.item(), self.progress.global_step
          )
          # Opcional: Logar a norma do delta atual e do delta de referência
          current_delta_norm, ref_delta_norm = self.delta_pattern.get_delta_norms()
          if current_delta_norm is not None:
            self.tensorboard.add_scalar("delta_pattern/current_total_delta_norm", current_delta_norm, self.progress.global_step)
          if ref_delta_norm is not None:
            self.tensorboard.add_scalar("delta_pattern/reference_delta_norm", ref_delta_norm, self.progress.global_step)

    # Retorna a loss final (por item do batch)
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
