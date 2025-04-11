from inspect import Parameter
import os
import time
import warnings
import torch
from torch import Tensor
from collections import deque
from typing import Iterable, Tuple, List, Dict, Union, Optional
from modules.util.NamedParameterGroup import NamedParameterGroupCollection

###############################################################################
# NOVO: Variável global para controlar o modo (scalar vs. batch)
###############################################################################
BATCH_MODE = True
"""
Se BATCH_MODE for False, comporta-se como no código original (usa .item()).
Se BATCH_MODE for True, as operações são feitas usando Tensores em lote (batch).
"""


class LossTracker:
    """
    Tracks and generates statistics for the latest N loss values for MSE, MAE, and log-cosh.
    It can use mean/std or median/MAD for statistics.
    """

    def __init__(self, window_size: int = 100, use_mad: bool = False) -> None:
        """
        Initializes the LossTracker.

        Args:
            window_size (int): The number of recent loss values to track.
            use_mad (bool): If True, use median and MAD instead of mean and std.
        """
        self.window_size: int = window_size
        self.use_mad: bool = use_mad

        # As filas continuarão existindo, mas podem armazenar floats ou Tensores
        self.mse_losses: deque = deque(maxlen=window_size)
        self.mae_losses: deque = deque(maxlen=window_size)
        self.log_cosh_losses: deque = deque(maxlen=window_size)

    def update(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor) -> None:
        """
        Updates the loss trackers with new loss values.

        Args:
            mse_loss (Tensor): The Mean Squared Error loss.
            mae_loss (Tensor): The Mean Absolute Error loss.
            log_cosh_loss (Tensor): The log-cosh loss.
        """
        if not BATCH_MODE:
            # Modo original (scalar)
            self.mse_losses.append(mse_loss.item())
            self.mae_losses.append(mae_loss.item())
            self.log_cosh_losses.append(log_cosh_loss.item())
        else:
            # Modo batch: podemos armazenar o Tensor inteiro,
            # ou apenas a média do batch, dependendo da sua necessidade.
            # Aqui, por exemplo, vamos armazenar o Tensor inteiro (detach)
            # para depois concatenar na hora de computar estatísticas.
            self.mse_losses.append(mse_loss.detach())
            self.mae_losses.append(mae_loss.detach())
            self.log_cosh_losses.append(log_cosh_loss.detach())

    def compute_stats(
        self, values_list: List[Union[float, Tensor]]
    ) -> Tuple[float, float]:
        """
        Computes statistics for a list of loss values.

        Args:
            values_list (List[float or Tensor]): The list of loss values (scalar ou Tensor).

        Returns:
            Tuple[float, float]: If use_mad is False, returns (mean, std).
                                 If use_mad is True, returns (median, MAD).
        """
        if len(values_list) < 2:
            # Evitar problemas no início do treinamento
            return 0.0, 1e-8

        if not BATCH_MODE:
            # Modo scalar original
            arr = torch.tensor(values_list, dtype=torch.float32)
        else:
            # Modo batch: "desempacota" Tensores em um único Tensor
            # Cada elemento de values_list pode ser shape [] (loss já reduzido)
            # ou [batch_size]. Aqui assumimos que cada item é [batch_size],
            # mas se for scalar, a concat ainda funciona com .view(-1).
            arr = torch.cat([x.view(-1) for x in values_list], dim=0)

        if not self.use_mad:
            mean_val = arr.mean()
            std_val = arr.std(unbiased=False)
            return mean_val.item(), max(std_val.item(), 1e-8)
        else:
            median_val = arr.median()
            abs_dev = torch.abs(arr - median_val)
            mad_val = abs_dev.median()
            return median_val.item(), max(mad_val.item(), 1e-8)

    def compute_z_scores(
        self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor
    ) -> Tuple[float, float, float]:
        """
        Computes z-scores for the given loss values.

        Args:
            mse_loss (Tensor): The Mean Squared Error loss.
            mae_loss (Tensor): The Mean Absolute Error loss.
            log_cosh_loss (Tensor): The log-cosh loss.

        Returns:
            Tuple[float, float, float]: The z-scores for MSE, MAE, and log-cosh losses.
        """
        # Primeiro, obtém as estatísticas de cada fila
        mse_center, mse_scale = self.compute_stats(self.mse_losses)
        mae_center, mae_scale = self.compute_stats(self.mae_losses)
        log_cosh_center, log_cosh_scale = self.compute_stats(self.log_cosh_losses)

        # Em modo batch, podemos tirar a média antes de calcular .item()
        # ou podemos calcular cada z-score como um vetor. Abaixo, fazemos
        # do jeito mais simples: tiramos a média do batch para manter
        # a coerência com o retorno float original.
        if not BATCH_MODE:
            mse_val = mse_loss.item()
            mae_val = mae_loss.item()
            log_cosh_val = log_cosh_loss.item()
        else:
            mse_val = mse_loss.mean().item()
            mae_val = mae_loss.mean().item()
            log_cosh_val = log_cosh_loss.mean().item()

        mse_z = (mse_val - mse_center) / mse_scale
        mae_z = (mae_val - mae_center) / mae_scale
        log_cosh_z = (log_cosh_val - log_cosh_center) / log_cosh_scale

        return mse_z, mae_z, log_cosh_z


class DynamicLossStrength:
    """
    Dynamically adjusts the weights of different loss components based on their z-scores.
    It can optionally use Exponential Moving Average (EMA) and applies a scheduling mechanism
    to prioritize different losses over the course of training.
    """

    def __init__(
        self,
        use_ema: bool = False,
        ema_decay: float = 0.9,
        outlier_threshold: float = 3.0,
        schedule_params: Dict[str, Dict[str, float]] = None,
    ) -> None:
        """
        Initializes the DynamicLossStrength.

        Args:
            use_ema (bool): Whether to use Exponential Moving Average for weights.
            ema_decay (float): Decay rate for EMA.
            outlier_threshold (float): Threshold to clamp z-scores.
            schedule_params (Dict[str, Dict[str, float]]): Scheduling parameters for each loss.
        """
        self.use_ema: bool = use_ema
        self.ema_decay: float = ema_decay
        self.outlier_threshold: float = outlier_threshold
        self.progress = None
        self.last_logged_delta_epoch = 0

        # Initialize scheduling parameters
        self.schedule_params: Dict[str, Dict[str, float]] = (
            self._initialize_schedule_params(schedule_params)
        )

        # EMA state
        self.ema_weights: Dict[str, float] = {"mse": 1.0, "mae": 1.0, "log_cosh": 1.0}
        self.initialized: bool = False

    def _initialize_schedule_params(
        self, schedule_params: Dict[str, Dict[str, float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Initializes the scheduling parameters.

        Args:
            schedule_params (Dict[str, Dict[str, float]], optional):
                User-provided scheduling parameters.

        Returns:
            Dict[str, Dict[str, float]]: Initialized scheduling parameters.
        """
        default_params = {
            "mae": {"start": 0.6, "end": 0.0},
            "mse": {"start": 0.2, "end": 0.6},
            "log_cosh": {"start": 0.2, "end": 0.4},
        }
        if schedule_params is not None:
            for loss_type, params in default_params.items():
                if loss_type in schedule_params:
                    default_params[loss_type].update(schedule_params[loss_type])
        return default_params

    def adjust_weights(
        self,
        mse_z: float,
        mae_z: float,
        log_cosh_z: float,
        config,
        progress,
    ) -> Tuple[float, float, float]:
        """
        Adjusts the weights for MSE, MAE, and log-cosh losses based on their z-scores.

        The adjustment process includes:
        1) Clamping z-scores to the outlier threshold.
        2) Inverting z-scores.
        3) Normalizing the inverted z-scores.
        4) Applying Exponential Moving Average (if enabled).
        5) Scheduling weight priorities over training epochs.

        Args:
            mse_z (float): Z-score for MSE loss.
            mae_z (float): Z-score for MAE loss.
            log_cosh_z (float): Z-score for log-cosh loss.
            config: Configuration object containing training parameters (e.g., total epochs).
            progress: Progress object containing current epoch information.

        Returns:
            Tuple[float, float, float]: The adjusted weights for MSE, MAE, and log-cosh losses.
        """
        # 1) Clamping z-scores
        z_scores = {
            "mse": min(abs(mse_z), self.outlier_threshold),
            "mae": min(abs(mae_z), self.outlier_threshold),
            "log_cosh": min(abs(log_cosh_z), self.outlier_threshold),
        }

        total_z = sum(z_scores.values())
        if total_z < 1e-8:
            total_z = 1.0

        # 2) Inverting z-scores
        inverted_z = {loss: (total_z - z) / total_z for loss, z in z_scores.items()}

        sum_inv = sum(inverted_z.values())
        if sum_inv < 1e-8:
            sum_inv = 1.0

        # 3) Normalizing inverted z-scores
        base_weights = {loss: inv_z / sum_inv for loss, inv_z in inverted_z.items()}

        # 4) Exponential Moving Average
        if self.use_ema:
            if not self.initialized:
                for loss in self.ema_weights:
                    self.ema_weights[loss] = base_weights[loss]
                self.initialized = True
            else:
                alpha = 1.0 - self.ema_decay
                for loss in self.ema_weights:
                    self.ema_weights[loss] = (1 - alpha) * self.ema_weights[
                        loss
                    ] + alpha * base_weights[loss]

            # Normaliza as weights do EMA
            sum_ema = sum(self.ema_weights.values())
            if sum_ema < 1e-8:
                sum_ema = 1.0
            normalized_ema_weights = {
                loss: w / sum_ema for loss, w in self.ema_weights.items()
            }
            base_weights = normalized_ema_weights

        # 5) Scheduler
        if config.epochs > 1:
            frac = progress.epoch / float(config.epochs - 1)
        else:
            frac = 0.0
        frac = max(0.0, min(frac, 1.0))

        scheduled_factors = {}
        for loss, params in self.schedule_params.items():
            scheduled_factors[loss] = (
                params["start"] * (1 - frac) + params["end"] * frac
            )

        # Multiplica cada base weight pelo fator do scheduler
        weighted_weights = {
            loss: base_weights[loss] * scheduled_factors[loss] for loss in base_weights
        }

        # Normaliza
        sum_weighted = sum(weighted_weights.values())
        if sum_weighted < 1e-8:
            sum_weighted = 1.0

        final_weights = {loss: w / sum_weighted for loss, w in weighted_weights.items()}

        # Retorna na ordem desejada
        return final_weights["mse"], final_weights["mae"], final_weights["log_cosh"]

    def maybe_log_deltas(self, tensorboard, delta_regularizer, progress):
        if not hasattr(self, "progress") or not self.progress:
            return

        if progress.epoch_step != 0:
            return

        if not hasattr(self, "last_logged_delta_epoch"):
            self.last_logged_delta_epoch = -1

        if progress.epoch == self.last_logged_delta_epoch:
            return

        self.last_logged_delta_epoch = self.progress.epoch

        current_norm, reference_norm = delta_regularizer.get_delta_norms()
        if current_norm is not None:
            tensorboard.add_scalar("delta/current_norm", current_norm, self.progress.global_step)
        if reference_norm is not None:
            tensorboard.add_scalar("delta/reference_norm", reference_norm, self.progress.global_step)

        print(f"[DeltaPattern] Epoch {progress.epoch} | Current Δ: {current_norm:.4f} | Ref Δ: {reference_norm:.4f}")


class DeltaPatternRegularizer:
    # INÍCIO ALTERAÇÃO: Recebe a coleção de parâmetros
    def __init__(self, param_collection: NamedParameterGroupCollection):
      self.param_collection: NamedParameterGroupCollection = param_collection
      # FIM ALTERAÇÃO
      self.initial_weights_run1: Dict[str, torch.Tensor] = {} # Pesos no início da Run 1 (para salvar)
      self.initial_weights_run2: Dict[str, torch.Tensor] = {} # Pesos no início da Run 2 (para calcular penalidade)
      self.reference_deltas: Dict[str, torch.Tensor] = {}   # Deltas carregados da Run 1
      self.current_total_delta_norm: Optional[float] = None # Cache da norma do delta atual
      self.reference_delta_norm: Optional[float] = None # Cache da norma do delta de referência

    # INÍCIO ALTERAÇÃO: Adaptado para iterar sobre a coleção
    def _iterate_params(self) -> Iterable[Tuple[str, Parameter]]:
      """Helper para iterar sobre todos os parâmetros com uma chave única."""
      for group in self.param_collection._NamedParameterGroupCollection__groups: # Acessa a lista privada
        if not group.parameters: # Pula grupos vazios
          continue
        group_device = group.parameters[0].device # Pega o device do primeiro parâmetro do grupo
        for i, param in enumerate(group.parameters):
          if param.requires_grad:
            key = f"{group.unique_name}_{i}"
            yield key, param, group_device # Retorna chave, param e device

    def capture_weights(self):
      """Captura os pesos iniciais (usado no início da Run 1)."""
      self.initial_weights_run1 = {} # Limpa antes de capturar
      for key, param, _ in self._iterate_params():
          self.initial_weights_run1[key] = param.detach().clone().cpu() # Armazena em CPU

      print(f"[DeltaPattern] Captured initial weights (Run 1) for {len(self.initial_weights_run1)} parameters across {len(self.param_collection._NamedParameterGroupCollection__groups)} groups.")
    # FIM ALTERAÇÃO

    # INÍCIO ALTERAÇÃO: Adaptado para iterar sobre a coleção
    def capture_initial_weights_run2(self):
      """Captura os pesos iniciais da Run 2 (usado para cálculo da penalidade)."""
      self.initial_weights_run2 = {} # Limpa antes de capturar
      for key, param, _ in self._iterate_params():
          self.initial_weights_run2[key] = param.detach().clone().cpu() # Armazena em CPU

      print(f"[DeltaPattern] Captured initial weights (Run 2) for {len(self.initial_weights_run2)} parameters across {len(self.param_collection._NamedParameterGroupCollection__groups)} groups.")
    # FIM ALTERAÇÃO

    # INÍCIO ALTERAÇÃO: Renomeado e ajustado para carregar
    def load_reference_pattern(self, pattern_path: str):
      """Carrega o padrão de delta de referência de um arquivo."""
      if not os.path.exists(pattern_path):
        warnings.warn(f"Reference delta pattern file not found: {pattern_path}", UserWarning)
        self.reference_deltas = {}
        return

      try:
        loaded_data = torch.load(pattern_path, map_location='cpu')
        if isinstance(loaded_data, dict):
          # Validação simples: verifica se as chaves parecem corretas
          if all(isinstance(k, str) and '_' in k for k in loaded_data.keys()):
            self.reference_deltas = loaded_data
            print(f"[DeltaPattern] Loaded reference delta pattern with {len(self.reference_deltas)} parameter deltas.")
            self.reference_delta_norm = self._calculate_total_norm(self.reference_deltas)
            print(f"[DeltaPattern] Reference delta total norm: {self.reference_delta_norm:.4f}")
          else:
            warnings.warn(f"Loaded delta pattern file has unexpected key format: {pattern_path}", UserWarning)
            self.reference_deltas = {}
        else:
          warnings.warn(f"Loaded delta pattern file is not a dictionary: {pattern_path}", UserWarning)
          self.reference_deltas = {}
      except Exception as e:
        warnings.warn(f"Failed to load reference delta pattern from {pattern_path}: {e}", UserWarning)
        self.reference_deltas = {}
  
    def _calculate_total_norm(self, weight_dict: Dict[str, Tensor]) -> float:
      """Calcula a norma L2 total sobre todos os tensores no dicionário."""
      if not weight_dict:
        return 0.0
      total_norm_sq = 0.0
      for name, tensor in weight_dict.items():
        total_norm_sq += torch.norm(tensor.to('cpu', dtype=torch.float32))**2
      return torch.sqrt(total_norm_sq).item()

	  # INÍCIO ALTERAÇÃO: Adaptado para iterar sobre a coleção
    def compute_penalty(self, lambda_weight: float) -> torch.Tensor:
      """Calcula a penalidade MSE entre o delta total atual e o delta de referência."""
      if not self.reference_deltas or not self.initial_weights_run2:
        # Tenta obter um device de qualquer parâmetro; se não houver, usa CPU
        try:
          device = next(self._iterate_params())[2]
        except StopIteration:
          device = 'cpu'
        return torch.tensor(0.0, device=device)

      total_penalty = torch.tensor(0.0, dtype=torch.float32) # Acumula em CPU
      current_deltas_for_norm = {} # Para calcular a norma atual
      num_params_matched = 0
      processed_keys = set() # Para evitar contar o mesmo parâmetro duas vezes se estiver em múltiplos grupos (improvável, mas seguro)

      for key, current_param, device in self._iterate_params():
        if key in processed_keys:
          continue

        if key in self.reference_deltas and key in self.initial_weights_run2:
          # Pega pesos iniciais e de referência (já estão em CPU)
          initial_weight_cpu = self.initial_weights_run2[key]
          ref_delta_cpu = self.reference_deltas[key]

          # Calcula o delta acumulado atual (current_param está no device original)
          current_delta = current_param.detach() - initial_weight_cpu.to(device)

          # Calcula a loss MSE (movendo ref_delta para o device correto)
          penalty_term = torch.nn.functional.mse_loss(
            current_delta.to(torch.float32),
            ref_delta_cpu.to(device, dtype=torch.float32)
          )
          # Acumula a penalidade (movendo para CPU para evitar múltiplas transferências GPU->CPU)
          total_penalty += penalty_term.cpu()

          current_deltas_for_norm[key] = current_delta.cpu() # Guarda delta atual em CPU para cálculo da norma
          num_params_matched += 1
          processed_keys.add(key)


      if num_params_matched > 0:
        self.current_total_delta_norm = self._calculate_total_norm(current_deltas_for_norm)
        # Média da penalidade pelos parâmetros comparados
        average_penalty = total_penalty / num_params_matched
      else:
        self.current_total_delta_norm = 0.0
        if not self.reference_deltas:
          warnings.warn("[DeltaPattern] Cannot compute penalty: Reference delta pattern not loaded.", UserWarning)
        elif not self.initial_weights_run2:
          warnings.warn("[DeltaPattern] Cannot compute penalty: Initial weights for Run 2 not captured.", UserWarning)
        else:
          warnings.warn("[DeltaPattern] No matching parameters found between current model and reference delta pattern.", UserWarning)
        average_penalty = torch.tensor(0.0) # Retorna 0 em CPU

      # Retorna a penalidade média ponderada, movida para o device do primeiro parâmetro (se existir)
      try:
        target_device = next(self._iterate_params())[2]
      except StopIteration:
        target_device = 'cpu' # Fallback para CPU

      return (lambda_weight * average_penalty).to(target_device)
    # FIM ALTERAÇÃO

    def get_delta_norms(self) -> Tuple[Optional[float], Optional[float]]:
      """Retorna a norma L2 total cacheada do delta atual e do delta de referência."""
      # (Inalterado)
      return self.current_total_delta_norm, self.reference_delta_norm

    # INÍCIO ALTERAÇÃO: Adaptado para iterar sobre a coleção
    def save_pattern(self, save_dir: str = "./training_pattern"):
      """Calcula o delta final da Run 1 e salva em um arquivo."""
      if not self.initial_weights_run1:
        print("[DeltaPattern] Error: Initial weights (Run 1) were not captured. Cannot save delta pattern.")
        return None

      print("[DeltaPattern] Calculating final delta pattern (Run 1)...")
      final_deltas: Dict[str, torch.Tensor] = {}
      final_weights_norm_sq = 0.0
      processed_keys = set()

      for key, param, device in self._iterate_params():
        if key in processed_keys:
          continue

        if key in self.initial_weights_run1:
          # Calcula o delta final em CPU
          initial_weight_cpu = self.initial_weights_run1[key]
          final_delta_cpu = param.detach().cpu() - initial_weight_cpu
          final_deltas[key] = final_delta_cpu # Armazena delta em CPU

          final_weights_norm_sq += torch.norm(param.detach().cpu().to(torch.float32))**2
          processed_keys.add(key)
        else:
          # Isso não deveria acontecer se capture_weights foi chamado corretamente
          warnings.warn(f"Parameter {key} found during save but not in initial weights capture.", UserWarning)


      initial_weights_norm = self._calculate_total_norm(self.initial_weights_run1)
      final_weights_norm = torch.sqrt(torch.tensor(final_weights_norm_sq)).item()
      final_delta_norm = self._calculate_total_norm(final_deltas)

      if not final_deltas:
        print("[DeltaPattern] Error: No matching parameters found to calculate final delta.")
        return None

      # (Lógica de salvar o arquivo e log inalterada)
      os.makedirs(save_dir, exist_ok=True)
      timestamp = time.strftime("%Y%m%d_%H%M%S")
      base_path = os.path.join(save_dir, f"pattern_{timestamp}")
      pattern_path = base_path + ".pt"
      log_path = base_path + ".txt"

      try:
        torch.save(final_deltas, pattern_path)
        print(f"[DeltaPattern] Saved final delta pattern to {pattern_path}")

        with open(log_path, "w") as f:
          f.write(f"Delta pattern saved: {pattern_path}\n")
          f.write(f"Timestamp: {timestamp}\n")
          f.write(f"Number of parameter deltas in pattern: {len(final_deltas)}\n")
          f.write(f"Number of parameter groups considered: {len(self.param_collection._NamedParameterGroupCollection__groups)}\n")
          f.write(f"Initial weights total norm (Run 1): {initial_weights_norm:.4f}\n")
          f.write(f"Final weights total norm (Run 1): {final_weights_norm:.4f}\n")
          f.write(f"Final delta total norm (Run 1): {final_delta_norm:.4f}\n\n")
          f.write("Delta details per parameter (Top 20 by Norm):\n")
          # Log apenas os top N deltas para evitar logs muito grandes
          sorted_deltas = sorted(final_deltas.items(), key=lambda item: item[1].norm().item(), reverse=True)
          for name, tensor in sorted_deltas[:20]:
            f.write(f"- {name}: {tensor.norm().item():.6f}\n")
          if len(sorted_deltas) > 20:
            f.write("...\n")


        return pattern_path
      except Exception as e:
        print(f"[DeltaPattern] Error saving delta pattern: {e}")
        return None
    # FIM ALTERAÇÃO
