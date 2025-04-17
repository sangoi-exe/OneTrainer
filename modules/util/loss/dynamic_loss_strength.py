import json
import traceback
from torch.nn import Parameter

# FIM ALTERAÇÃO
import os
import time
import warnings
import torch
from torch import Tensor
from collections import deque
from safetensors.torch import load_file

# INÍCIO ALTERAÇÃO: Ajustar imports e tipos
from typing import Iterable, Tuple, List, Dict, Union, Optional, TYPE_CHECKING

from modules.util.NamedParameterGroup import NamedParameterGroup

# Usar TYPE_CHECKING para evitar importação circular se NamedParameterGroupCollection estiver em outro módulo
if TYPE_CHECKING:
    from modules.util.NamedParameterGroup import NamedParameterGroupCollection
# FIM ALTERAÇÃO

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
        # INÍCIO ALTERAÇÃO: armazenar sempre na CPU para evitar consumo de VRAM
        if not BATCH_MODE:
            # Modo original (scalar)
            self.mse_losses.append(mse_loss.item())
            self.mae_losses.append(mae_loss.item())
            self.log_cosh_losses.append(log_cosh_loss.item())
        else:
            # Modo batch: armazena o tensor inteiro mas já movido para a CPU
            self.mse_losses.append(mse_loss.detach().cpu())
            self.mae_losses.append(mae_loss.detach().cpu())
            self.log_cosh_losses.append(log_cosh_loss.detach().cpu())
        # FIM ALTERAÇÃO


    def compute_stats(self, values_list: List[Union[float, Tensor]]) -> Tuple[float, float]:
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
            arr = torch.tensor(values_list, dtype=torch.float32, device="cpu")
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

    def compute_z_scores(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor) -> Tuple[float, float, float]:
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
        self.schedule_params: Dict[str, Dict[str, float]] = self._initialize_schedule_params(schedule_params)

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
                    self.ema_weights[loss] = (1 - alpha) * self.ema_weights[loss] + alpha * base_weights[loss]

            # Normaliza as weights do EMA
            sum_ema = sum(self.ema_weights.values())
            if sum_ema < 1e-8:
                sum_ema = 1.0
            normalized_ema_weights = {loss: w / sum_ema for loss, w in self.ema_weights.items()}
            base_weights = normalized_ema_weights

        # 5) Scheduler
        if config.epochs > 1:
            frac = progress.epoch / float(config.epochs - 1)
        else:
            frac = 0.0
        frac = max(0.0, min(frac, 1.0))

        scheduled_factors = {}
        for loss, params in self.schedule_params.items():
            scheduled_factors[loss] = params["start"] * (1 - frac) + params["end"] * frac

        # Multiplica cada base weight pelo fator do scheduler
        weighted_weights = {loss: base_weights[loss] * scheduled_factors[loss] for loss in base_weights}

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
    def __init__(self, model: torch.nn.Module):
        try:
            collection = model.parameter_collection
        except AttributeError:
            raise ValueError(
                "Modelo não possui 'parameter_collection' — "
                "verifique se você atribuiu antes de instanciar."
            )
        self.model = model
        self.param_collection: "NamedParameterGroupCollection" = collection
        self.delta_log_by_module: Dict[str, Dict[str, float]] = {}
        self.initial_weights_run1: Dict[str, torch.Tensor] = (
            {}
        )  # Pesos no início da Run 1 (para salvar) - Em CPU
        self.initial_weights_run2: Dict[str, torch.Tensor] = (
            {}
        )  # Pesos no início da Run 2 (para calcular penalidade) - Em CPU
        self.reference_deltas: Dict[str, torch.Tensor] = (
            {}
        )  # Deltas carregados da Run 1 - Em CPU
        self.current_total_delta_norm: Optional[float] = (
            None  # Cache da norma L2 do delta atual
        )
        self.reference_delta_norm: Optional[float] = (
            None  # Cache da norma L2 do delta de referência
        )

    def _iterate_params(self) -> Iterable[Tuple[str, torch.Tensor, torch.device]]:
        """
        Itera diretamente sobre o state_dict do unet_lora e retorna todos os pesos reais com nome completo.
        NÃO depende de .named_parameters() para evitar perder buffers como alpha/dora_scale.
        """
        for name, tensor in self.model.unet_lora.state_dict().items():
            if not isinstance(tensor, torch.Tensor):
                continue
            yield name, tensor, tensor.device

    def capture_weights(self):
        """Captura os pesos iniciais (usado no início da Run 1) e armazena em CPU."""
        self.initial_weights_run1 = {}  # Limpa antes de capturar
        count = 0
        try:
            for key, param, _ in self._iterate_params():
                self.initial_weights_run1[key] = param.detach().cpu()  # Armazena em CPU
                count += 1
        except Exception as e:
            print(f"[DeltaPattern] Erro durante capture_weights: {e}")
            traceback.print_exc()
            raise  # Re-levanta a exceção para indicar falha

        if count == 0:
            print("[DeltaPattern] capture_weights não encontrou parâmetros treináveis.")
        else:
            print(
                f"[DeltaPattern] Capturou pesos iniciais (Run 1) para {count} parâmetros treináveis."
            )
            # Opcional: Calcular e logar norma inicial aqui se desejado
            # initial_norm = self._calculate_total_norm(self.initial_weights_run1)
            # print(f"[DeltaPattern] Norma L2 total dos pesos iniciais (Run 1): {initial_norm:.4f}")

    def capture_initial_weights_run2(self):
        """Captura os pesos iniciais da Run 2 (usado para cálculo da penalidade) e armazena em CPU."""
        self.initial_weights_run2 = {}  # Limpa antes de capturar
        count = 0
        try:
            for key, param, _ in self._iterate_params():
                self.initial_weights_run2[key] = (
                    param.detach().clone().cpu()
                )  # Armazena em CPU
                count += 1
        except Exception as e:
            print(f"[DeltaPattern] Erro durante capture_initial_weights_run2: {e}")
            traceback.print_exc()
            raise

        if count == 0:
            print(
                "[DeltaPattern] capture_initial_weights_run2 não encontrou parâmetros treináveis."
            )
        else:
            print(
                f"[DeltaPattern] Capturou pesos iniciais (Run 2) para {count} parâmetros treináveis."
            )
            # Opcional: Calcular e logar norma inicial aqui se desejado
            # initial_norm_run2 = self._calculate_total_norm(self.initial_weights_run2)
            # print(f"[DeltaPattern] Norma L2 total dos pesos iniciais (Run 2): {initial_norm_run2:.4f}")

    def load_reference_pattern(self, pattern_path: str):
        """Carrega o padrão de delta de referência de um arquivo JSON."""
        self.reference_deltas = {}
        self.reference_delta_norm = None

        if not os.path.isfile(pattern_path):
            print(f"[DeltaPattern] Arquivo JSON não encontrado: {pattern_path}")
            return

        try:
            with open(pattern_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            flat_dict = {}
            for epoch_key, metrics in json_data.items():
                for param_name, value in metrics.items():
                    full_key = f"{epoch_key}/{param_name}"
                    flat_dict[full_key] = torch.tensor(
                        value, dtype=torch.float32, device="cpu"
                    )

            if flat_dict:
                self.reference_deltas = flat_dict
                self.reference_delta_norm = self._calculate_total_norm(
                    self.reference_deltas
                )
                epoch_numbers = [
                    int(k.split("/")[0].replace("epoch_", ""))
                    for k in flat_dict.keys()
                    if k.startswith("epoch_")
                ]
                self.max_epoch_loaded = max(epoch_numbers) if epoch_numbers else None
                print(
                    f"[DeltaPattern] JSON '{pattern_path}' carregado com {len(flat_dict)} deltas."
                )
                print(
                    f"[DeltaPattern] Último epoch registrado no JSON: {self.max_epoch_loaded}"
                )
                print(
                    f"[DeltaPattern] Norma L2 total do delta de referência: {self.reference_delta_norm:.4f}"
                )
            else:
                print("[DeltaPattern] JSON estava vazio ou mal formatado.")
        except Exception as e:
            print(f"[DeltaPattern] Falha ao carregar JSON de deltas: {e}")
            traceback.print_exc()
            self.reference_deltas = {}

    def compute_penalty(self, lambda_weight: float) -> torch.Tensor:
        """
        Calcula a penalidade MSE entre o delta acumulado atual (current - initial_run2)
        e o delta de referência (carregado da Run 1).
        Retorna um tensor escalar no device do primeiro parâmetro do modelo.
        """
        try:
            # INÍCIO ALTERAÇÃO: Obter device e dtype do modelo
            first_param = next(self.model.parameters())
            target_device = first_param.device
            target_dtype = first_param.dtype  # Assume bf16 se o modelo estiver em bf16
            # FIM ALTERAÇÃO
        except StopIteration:
            target_device = torch.device("cpu")
            target_dtype = torch.float32  # Fallback dtype
            print(
                "[DeltaPattern] Não foi possível determinar o device/dtype do modelo. Usando CPU/float32.",
                UserWarning,
            )

        if hasattr(self, "max_epoch_loaded") and hasattr(self.model, "train_progress"):
            current_epoch = self.model.train_progress.epoch
            if (
                self.max_epoch_loaded is not None
                and current_epoch > self.max_epoch_loaded
            ):
                print(
                    f"[DeltaPattern] Aviso: Epoch atual ({current_epoch}) excede o máximo registrado no padrão ({self.max_epoch_loaded}). Penalidade será ignorada."
                )

        if not self.reference_deltas or not self.initial_weights_run2:
            self.current_total_delta_norm = None
            return torch.tensor(
                0.0, device=target_device, dtype=target_dtype
            )  # Usa dtype do modelo

        try:
            initial_run2_device = {
                k: v.to(device=target_device, dtype=target_dtype, non_blocking=True)
                for k, v in self.initial_weights_run2.items()
            }
            ref_delta_device = {
                k: v.to(device=target_device, dtype=target_dtype, non_blocking=True)
                for k, v in self.reference_deltas.items()
            }
        except Exception as e:
            print(
                f"[DeltaPattern] Erro ao mover/castar pesos iniciais/referência para {target_device}/{target_dtype}: {e}"
            )
            traceback.print_exc()
            return torch.tensor(0.0, device=target_device, dtype=target_dtype)

        # INÍCIO ALTERAÇÃO: Acumular penalidade e deltas na GPU
        total_penalty_gpu = torch.tensor(0.0, dtype=torch.float32, device=target_device)
        total_elements = 0
        current_deltas_gpu = {}  # Guarda deltas atuais na GPU para cálculo da norma L2
        # FIM ALTERAÇÃO
        num_params_matched = 0

        try:
            # REMOVIDO: Cache redundante que estava dentro do try original
            # target_device = next(self.model.parameters()).device
            # initial_run2_device = { ... }
            # ref_delta_device = { ... }

          current_vec = torch.nn.utils.parameters_to_vector(
              [p.detach().to(target_dtype) for _, p, _ in self._iterate_params()]
          )
          init_vec = torch.nn.utils.parameters_to_vector(
              [self.initial_weights_run2[k].to(target_dtype) for k, _, _ in self._iterate_params()]
          )
          ref_vec = torch.nn.utils.parameters_to_vector(
              [self.reference_deltas[k] for k in self.reference_deltas]
          ).to(target_dtype)

          current_delta = current_vec - init_vec
          penalty = torch.nn.functional.mse_loss(current_delta.float(), ref_vec.float())
          self.current_total_delta_norm = torch.norm(current_delta.float(), p=2).item()
          final_penalty = (lambda_weight * penalty).to(dtype=target_dtype)
          return final_penalty

        except Exception as e:
            print(
                f"[DeltaPattern] Erro durante compute_penalty ao iterar parâmetros: {e}"
            )
            traceback.print_exc()
            self.current_total_delta_norm = None
            return torch.tensor(0.0, device=target_device, dtype=target_dtype)

        if num_params_matched > 0:
            # INÍCIO ALTERAÇÃO: Calcular norma L2 na GPU e depois mover para CPU
            # Calcula a norma L2 total do delta *atual* usando os deltas na GPU
            # self.current_total_delta_norm = self._calculate_total_norm(current_deltas_for_norm) # Modificado
            current_norm_sq_gpu = torch.tensor(
                0.0, dtype=torch.float64, device=target_device
            )  # float64 para precisão
            for delta_gpu in current_deltas_gpu.values():
                delta_norm_sq = (
                    torch.norm(delta_gpu.to(torch.float32), p=2).pow(2).double()
                )
                current_norm_sq_gpu += delta_norm_sq
            self.current_total_delta_norm = torch.sqrt(
                current_norm_sq_gpu
            ).item()  # .item() move escalar final para CPU
            # FIM ALTERAÇÃO

            # Calcula a penalidade média pelos parâmetros comparados (ainda na GPU)
            average_penalty_gpu = total_penalty_gpu / total_elements
        else:
            self.current_total_delta_norm = 0.0
            if self.reference_deltas and self.initial_weights_run2:
                print(
                    "[DeltaPattern] Nenhum parâmetro correspondente encontrado...",
                    UserWarning,
                )  # Mensagem existente
            average_penalty_gpu = torch.tensor(
                0.0, device=target_device, dtype=torch.float32
            )  # Penalidade média 0 na GPU

        # Retorna a penalidade média ponderada, no device e dtype originais do modelo
        final_penalty = (lambda_weight * average_penalty_gpu).to(
            dtype=target_dtype
        )  # Cast final para dtype do modelo
        return final_penalty

    # INÍCIO ALTERAÇÃO: Modificar _calculate_total_norm para aceitar device
    def _calculate_total_norm(
        self, weight_dict: Dict[str, Tensor], device: torch.device = torch.device("cpu")
    ) -> float:
        """Calcula a norma L2 total sobre todos os tensores no dicionário."""
        if not weight_dict:
            return 0.0
        # Usa float64 para precisão na soma, no device especificado
        total_norm_sq = torch.tensor(0.0, dtype=torch.float64, device=device)
        for tensor in weight_dict.values():
            # Garante que o tensor está no device correto e calcula a norma quadrada
            # Usa .float() para norm que pode não suportar bf16 diretamente
            total_norm_sq += (
                torch.norm(tensor.to(device=device, non_blocking=True).float(), p=2)
                .pow(2)
                .double()
            )
        return torch.sqrt(total_norm_sq).item()  # .item() move para CPU

    # FIM ALTERAÇÃO

    def get_delta_norms(self) -> Tuple[Optional[float], Optional[float]]:
        """Retorna a norma L2 total cacheada do delta atual e do delta de referência."""
        return self.current_total_delta_norm, self.reference_delta_norm

    def log_group_deltas(self, epoch: int):
        """
        Salva norma L2 do delta para cada grupo (agrupados por prefixo simples) no epoch atual.
        Usa self.initial_weights_run1 como base.
        """
        if not self.initial_weights_run1:
            print("[DeltaPattern] log_group_deltas: pesos iniciais não capturados.")
            return

        epoch_key = f"epoch_{epoch}"
        self.delta_log_by_module[epoch_key] = {}
        # do_tensorboard = hasattr(self, "tensorboard") and self.tensorboard is not None

        # INÍCIO ALTERAÇÃO: iterar sobre parâmetros reais e agrupar por prefixo
        group_norms: Dict[str, float] = {}
        param_counts: Dict[str, int] = {}

        for name, current_param, _ in self._iterate_params():
            if name not in self.initial_weights_run1:
                continue

            initial_weight = self.initial_weights_run1[name].to(
                dtype=torch.float32, device=current_param.device
            )
            delta = current_param.detach().to(dtype=torch.float32) - initial_weight

            # Define o prefixo de agrupamento (ajuste aqui para granularidade desejada)
            prefix = name.split(".")[
                0
            ]  # ou use um parser mais detalhado se quiser algo como 'up_blocks_2_resnets_1'

            group_norms.setdefault(prefix, 0.0)
            param_counts.setdefault(prefix, 0)

            group_norms[prefix] += torch.norm(delta, p=2).pow(2).item()
            param_counts[prefix] += 1

        for prefix, norm_sq in group_norms.items():
            count = param_counts[prefix]
            delta_norm = norm_sq**0.5 if count > 0 else 0.0
            self.delta_log_by_module[epoch_key][prefix] = delta_norm
        # FIM ALTERAÇÃO
        # if do_tensorboard:
        #     self.tensorboard.add_scalar(f"delta/by_group/{prefix}", delta_norm, epoch)

        print(f"[DeltaPattern] Deltas logados para epoch {epoch_key}.")

    def save_group_deltas(
        self, path: str = "./training_pattern/delta_log_by_module.json"
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.delta_log_by_module, f, indent=2)
        print(f"[DeltaPattern] Delta por grupo salvo em {path}")