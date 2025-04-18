import json
import traceback
import os
import time
import warnings
import torch
from torch import Tensor
from collections import deque
from safetensors.torch import load_file

from typing import Iterable, Tuple, List, Dict, Union, Optional, TYPE_CHECKING

from modules.util.NamedParameterGroup import NamedParameterGroup

if TYPE_CHECKING:
    from modules.util.NamedParameterGroup import NamedParameterGroupCollection


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

        # Modo batch: armazena o tensor inteiro mas já movido para a CPU
        self.mse_losses.append(mse_loss.detach().cpu())
        self.mae_losses.append(mae_loss.detach().cpu())
        self.log_cosh_losses.append(log_cosh_loss.detach().cpu())

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

        # Modo batch: "desempacota" Tensores em um único Tensor
        # Cada elemento de values_list pode ser shape [] (loss já reduzido)
        # ou [batch_size]. Aqui assumimos que cada item é [batch_size],
        # mas se for scalar, a concat ainda funciona com .view(-1).
        arr = torch.tensor(values_list, dtype=torch.float32, device="cpu")

        if not self.use_mad:
            mean_val = arr.mean()
            std_val = arr.std(unbiased=False)
            # .item() será chamado apenas quando for realmente necessário obter um float na CPU
            return mean_val, torch.clamp(
                std_val, min=1e-8
            )  # Retorna tensores escalares
        else:
            median_val = arr.median()
            abs_dev = torch.abs(arr - median_val)
            mad_val = abs_dev.median().values
            return median_val, torch.clamp(
                mad_val, min=1e-8
            )  # Retorna tensores escalares
    @torch.no_grad()
    def compute_z_scores(
        self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor
    ) -> Union[Tuple[float, float, float], Tuple[Tensor, Tensor, Tensor]]:
        """
        Computes z-scores for the given loss values.

        Args:
            mse_loss (Tensor): The Mean Squared Error loss.
            mae_loss (Tensor): The Mean Absolute Error loss.
            log_cosh_loss (Tensor): The log-cosh loss.

        Returns:
            Union[Tuple[float, float, float], Tuple[Tensor, Tensor, Tensor]]:
            The z-scores for MSE, MAE, and log-cosh losses.
            Tensors (shape like input losses).
        """

        mse_center, mse_scale = self.compute_stats(list(self.mse_losses))
        mae_center, mae_scale = self.compute_stats(list(self.mae_losses))
        log_cosh_center, log_cosh_scale = self.compute_stats(list(self.log_cosh_losses))

        mse_z = (mse_loss - mse_center) / mse_scale
        mae_z = (mae_loss - mae_center) / mae_scale
        log_cosh_z = (log_cosh_loss - log_cosh_center) / log_cosh_scale

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
    @torch.no_grad()
    def adjust_weights(
        self,
        mse_z: Union[float, Tensor],
        mae_z: Union[float, Tensor],
        log_cosh_z: Union[float, Tensor],
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
            mse_z (Union[float, Tensor]): Z-score(s) for MSE loss.
            mae_z (Union[float, Tensor]): Z-score(s) for MAE loss.
            log_cosh_z (float): Z-score for log-cosh loss.
            config: Configuration object containing training parameters (e.g., total epochs).
            progress: Progress object containing current epoch information.

        Returns:
            Tuple[float, float, float]: The adjusted weights for MSE, MAE, and log-cosh losses.
        """
        # 1) Clamping z-scores
        _abs = torch.abs
        _clamp_max = lambda t, val: torch.clamp(t, max=val)
        _sum = lambda d: torch.stack(list(d.values())).sum(dim=0)
        _mean = lambda t: t.mean()  # Usado para obter pesos escalares finais
        is_tensor_mode = True

        # 1) Clamping z-scores (valor absoluto clampado)
        # Clampa o valor *absoluto* e depois o usa. Ou clampa o z original entre -thresh e +thresh?
        # O código original fazia min(abs(z), threshold). Vamos manter isso.
        clamped_abs_z = {
            "mse": _clamp_max(_abs(mse_z), self.outlier_threshold),
            "mae": _clamp_max(_abs(mae_z), self.outlier_threshold),
            "log_cosh": _clamp_max(_abs(log_cosh_z), self.outlier_threshold),
        }

        # 2) Invertendo z-scores relativos
        # A lógica original era: inv_z = (total_abs_z - abs_z) / total_abs_z
        # Isso dá peso maior para quem tem z-score absoluto menor.
        total_abs_z = _sum(clamped_abs_z)
        # Adicionar epsilon para evitar divisão por zero (importante para tensores)
        epsilon = 1e-8
        total_abs_z = total_abs_z + epsilon  # Broadcasting se for tensor

        inverted_z = {
            loss: (total_abs_z - z) / total_abs_z for loss, z in clamped_abs_z.items()
        }

        # 3) Normalizando pesos base
        # sum_inv_z = (total-z1)/total + (total-z2)/total + (total-z3)/total
        #           = (3*total - (z1+z2+z3))/total = (3*total - total)/total = 2*total / total = 2 (se total > 0)
        # Portanto, a soma dos inverted_z deveria ser 2 (ou próximo disso devido ao clamp e epsilon).
        # Normalizar dividindo pela soma garante que os pesos somem 1.
        sum_inv_z = _sum(inverted_z) + epsilon
        base_weights = {loss: inv_z / sum_inv_z for loss, inv_z in inverted_z.items()}

        # 4) Exponential Moving Average
        if self.use_ema:
            if not self.initialized:
                # Inicializa com os pesos atuais (podem ser tensores ou floats)
                for loss in self.ema_weights:
                    self.ema_weights[loss] = base_weights[loss]
            else:
                alpha = 1.0 - self.ema_decay
                for loss in self.ema_weights:
                    self.ema_weights[loss] = (1 - alpha) * self.ema_weights[
                        loss
                    ] + alpha * base_weights[loss]
                # self.initialized = True # Cuidado: Se base_weights for Tensor, ema_weights vira Tensor
            # Normaliza as weights do EMA
            sum_ema = sum(self.ema_weights.values())

            # Se ema_weights for Tensor, sum_ema será Tensor. Adicionar epsilon.
            sum_ema = sum_ema + epsilon
            normalized_ema_weights = {
                loss: w / sum_ema for loss, w in self.ema_weights.items()
            }  # Broadcasting se Tensor
            base_weights = normalized_ema_weights
            # Se inicializou com tensores, aqui base_weights são tensores
            self.initialized = (
                True  # Marcar como inicializado após a primeira atualização/leitura
            )

        # 5) Scheduler
        if config.epochs > 1:
            frac = progress.epoch / float(config.epochs - 1)
        else:
            frac = 0.0
        frac = max(0.0, min(frac, 1.0))

        scheduled_factors = {}
        for loss, params in self.schedule_params.items():
            # params são floats, o resultado é float
            scheduled_factors[loss] = (
                params["start"] * (1 - frac) + params["end"] * frac
            )

        # Multiplica cada base weight pelo fator do scheduler
        # Se base_weights for Tensor, o fator float faz broadcasting
        weighted_weights = {
            loss: base_weights[loss] * scheduled_factors[loss] for loss in base_weights
        }

        # Normaliza final
        sum_weighted = _sum(weighted_weights) + epsilon
        final_weights = {loss: w / sum_weighted for loss, w in weighted_weights.items()}

        # 6) Redução para escalar (se necessário)
        # A loss final espera pesos escalares. Se estávamos no modo tensor,
        # pegamos a média dos pesos no batch.
        if is_tensor_mode:
            final_weights_scalar = {
                loss: float(torch.mean(w)) for loss, w in final_weights.items()
            }
        else:
            final_weights_scalar = final_weights  # Já são floats

        # Retorna na ordem desejada
        return (
            final_weights_scalar["mse"],
            final_weights_scalar["mae"],
            final_weights_scalar["log_cosh"],
        )

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
            tensorboard.add_scalar(
                "delta/current_norm", current_norm, self.progress.global_step
            )
        if reference_norm is not None:
            tensorboard.add_scalar(
                "delta/reference_norm", reference_norm, self.progress.global_step
            )

        print(
            f"[DeltaPattern] Epoch {progress.epoch} | Current Δ: {current_norm:.4f} | Ref Δ: {reference_norm:.4f}"
        )


class DeltaPatternRegularizer:
    def __init__(
        self, model: torch.nn.Module, param_collection: "NamedParameterGroupCollection", penalty_metric: str = "cosine"
    ):
        if param_collection is None:
            raise ValueError(
                "param_collection não pode ser None para DeltaPatternRegularizer."
            )
        self.model = model
        self.param_collection: "NamedParameterGroupCollection" = param_collection
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
        self.penalty_metric: str = penalty_metric.lower()
        if self.penalty_metric not in {"mse", "cosine"}:
            raise ValueError("penalty_metric deve ser 'mse' ou 'cosine'")

    def _iterate_params(self) -> Iterable[Tuple[str, torch.Tensor, torch.device]]:
        """Iterates through tensors within the state_dicts of relevant LoRA wrappers."""
        wrappers_to_check = [
            getattr(self.model, "unet_lora", None),
        ]
        processed_keys = (
            set()
        )  # Evita processar a mesma chave de diferentes wrappers (improvável mas seguro)

        for wrapper in wrappers_to_check:
            if wrapper is None:
                continue
            try:
                # state_dict() do wrapper deve retornar chaves de módulos reais/dummies
                wrapper_state_dict = wrapper.state_dict()
                for key, tensor in wrapper_state_dict.items():
                    # Verifica se é um tensor e ainda não foi processado
                    if isinstance(tensor, torch.Tensor) and key not in processed_keys:
                        # Assume que todos os tensores no state_dict são relevantes
                        # Retorna tensor destacado (detach) para evitar problemas de grafo
                        yield key, tensor.detach(), tensor.device
                        processed_keys.add(key)
            except Exception as e:
                print(
                    f"[DeltaPattern] Erro ao iterar state_dict para wrapper {getattr(wrapper, 'prefix', 'Unknown')}: {e}"
                )

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
        Calcula penalidade entre delta atual (por módulo) e delta de referência.
        Métrica definida em `self.penalty_metric` ("mse" ou "cosine").
        """
        import warnings

        # ── descobrir device/dtype ───────────────────────────────────────────
        try:
            first_param = next(iter(self.param_collection.parameters()))
            target_device, target_dtype = first_param.device, first_param.dtype
        except StopIteration:
            target_device, target_dtype = torch.device("cpu"), torch.float32
            warnings.warn("[DeltaPattern] Parâmetros vazios; usando CPU/float32.")

        # ── early‑exit se epoch > padrão carregado ───────────────────────────
        if hasattr(self, "max_epoch_loaded") and hasattr(self.model, "train_progress"):
            if self.max_epoch_loaded is not None and self.model.train_progress.epoch > self.max_epoch_loaded:
                return torch.tensor(0.0, device=target_device, dtype=target_dtype)

        # ── pré‑condições ────────────────────────────────────────────────────
        if not self.reference_deltas or not self.initial_weights_run2:
            self.current_total_delta_norm = None
            return torch.tensor(0.0, device=target_device, dtype=target_dtype)

        try:
            # ── preparar dict referência → módulo ────────────────────────────
            ref_mod: Dict[str, torch.Tensor] = {}
            for full_key, val in self.reference_deltas.items():
                module_key = full_key.split("/", 1)[-1]  # remove "epoch_x/" se houver
                ref_mod[module_key] = val.to(
                    device=target_device, dtype=target_dtype, non_blocking=True
                )

            # ── deltas atuais por módulo ─────────────────────────────────────
            cur_mod = self._get_current_module_deltas(target_device, target_dtype)

            common_keys = [k for k in ref_mod if k in cur_mod]
            if not common_keys:
                self.current_total_delta_norm = 0.0
                return torch.tensor(0.0, device=target_device, dtype=target_dtype)

            ref_vec = torch.stack([ref_mod[k].float() for k in common_keys])
            cur_vec = torch.stack([cur_mod[k].float() for k in common_keys])

            # ─────────── INÍCIO ALTERAÇÃO CHATGPT ───────────
            if self.penalty_metric == "cosine":
                cos_sim = torch.nn.functional.cosine_similarity(cur_vec, ref_vec, dim=0, eps=1e-8)
                penalty = 1.0 - cos_sim  # distância angular
            else:  # "mse"
                penalty = torch.nn.functional.mse_loss(cur_vec, ref_vec)
            # ──────────── FIM ALTERAÇÃO CHATGPT ────────────

            self.current_total_delta_norm = torch.norm(cur_vec, p=2).item()
            return (lambda_weight * penalty).to(dtype=target_dtype)

        except Exception as e:
            print(f"[DeltaPattern] Erro compute_penalty: {e}")
            traceback.print_exc()
            self.current_total_delta_norm = None
            return torch.tensor(0.0, device=target_device, dtype=target_dtype)



    def _calculate_total_norm(
        self, weight_dict: Dict[str, Tensor], device: torch.device = torch.device("cpu")
    ) -> float:
        """Calcula a norma L2 total sobre todos os tensores no dicionário."""
        if not weight_dict:
            return 0.0
        # Usa float64 para precisão na soma, no device especificado
        total_norm_sq = torch.tensor(0.0, dtype=torch.float32, device=device)
        for tensor in weight_dict.values():
            # Garante que o tensor está no device correto e calcula a norma quadrada
            # Usa .float() para norm que pode não suportar bf16 diretamente
            total_norm_sq += (
                torch.norm(tensor.to(device=device, non_blocking=True).float(), p=2)
                .pow(2)
                .double()
            )
        return torch.sqrt(total_norm_sq).item()  # .item() move para CPU

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
            ]

            group_norms.setdefault(prefix, 0.0)
            param_counts.setdefault(prefix, 0)

            group_norms[prefix] += torch.norm(delta, p=2).pow(2).item()
            param_counts[prefix] += 1

        for prefix, norm_sq in group_norms.items():
            count = param_counts[prefix]
            delta_norm = norm_sq**0.5 if count > 0 else 0.0
            self.delta_log_by_module[epoch_key][prefix] = delta_norm
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

    # ────────────── INÍCIO INSERÇÃO CHATGPT ──────────────
    def _get_current_module_deltas(
        self, target_device: torch.device, target_dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula e retorna {prefixo_módulo: norma_L2_delta_atual} no device/dtype alvos.
        """
        deltas_sq: Dict[str, torch.Tensor] = {}

        for name, current_param, _ in self._iterate_params():
            if name not in self.initial_weights_run2:
                continue
            prefix = name.split(".")[0]

            initial = self.initial_weights_run2[name].to(
                device=target_device, dtype=target_dtype, non_blocking=True
            )
            delta = (current_param.detach().to(dtype=target_dtype) - initial).float()
            deltas_sq.setdefault(
                prefix,
                torch.tensor(0.0, device=target_device, dtype=torch.float32),
            )
            deltas_sq[prefix] += torch.norm(delta, p=2).pow(2)

        return {k: torch.sqrt(v) for k, v in deltas_sq.items()}
    # ─────────────── FIM INSERÇÃO CHATGPT ───────────────

