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
        # Modo batch: podemos armazenar o Tensor inteiro,
        # ou apenas a média do batch, dependendo da sua necessidade.
        # Aqui, por exemplo, vamos armazenar o Tensor inteiro (detach)
        # para depois concatenar na hora de computar estatísticas.
        self.mse_losses.append(mse_loss)
        self.mae_losses.append(mae_loss)
        self.log_cosh_losses.append(log_cosh_loss)

        """
        SE DER OOM, USAR MEAN NESSA BOSTA
        """
        # self.mse_losses.append(mse_loss.detach().mean())
        # self.mae_losses.append(mae_loss.detach().mean())
        # self.log_cosh_losses.append(log_cosh_loss.detach().mean())

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

        # Modo batch: "desempacota" Tensores em um único Tensor
        # Cada elemento de values_list pode ser shape [] (loss já reduzido)
        # ou [batch_size]. Aqui assumimos que cada item é [batch_size],
        # mas se for scalar, a concat ainda funciona com .view(-1).
        arr = torch.cat([x.view(-1) for x in values_list], dim=0)

        if not self.use_mad:
            mean_val = arr.mean()
            std_val = arr.std(unbiased=False)
            # .item() será chamado apenas quando for realmente necessário obter um float na CPU
            return mean_val, torch.clamp(std_val, min=1e-8)  # Retorna tensores escalares
        else:
            median_val = arr.median()
            abs_dev = torch.abs(arr - median_val)
            mad_val = abs_dev.median().values
            return median_val, torch.clamp(mad_val, min=1e-8)  # Retorna tensores escalares

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

        Args: # INÍCIO ALTERAÇÃO: Atualizar docstring
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

        inverted_z = {loss: (total_abs_z - z) / total_abs_z for loss, z in clamped_abs_z.items()}

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
                    self.ema_weights[loss] = (1 - alpha) * self.ema_weights[loss] + alpha * base_weights[loss]
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
            self.initialized = True  # Marcar como inicializado após a primeira atualização/leitura

        # 5) Scheduler
        if config.epochs > 1:
            frac = progress.epoch / float(config.epochs - 1)
        else:
            frac = 0.0
        frac = max(0.0, min(frac, 1.0))

        scheduled_factors = {}
        for loss, params in self.schedule_params.items():
            # params são floats, o resultado é float
            scheduled_factors[loss] = params["start"] * (1 - frac) + params["end"] * frac

        # Multiplica cada base weight pelo fator do scheduler
        # Se base_weights for Tensor, o fator float faz broadcasting
        weighted_weights = {loss: base_weights[loss] * scheduled_factors[loss] for loss in base_weights}

        # Normaliza final
        sum_weighted = _sum(weighted_weights) + epsilon
        final_weights = {loss: w / sum_weighted for loss, w in weighted_weights.items()}

        # 6) Redução para escalar (se necessário)
        # A loss final espera pesos escalares. Se estávamos no modo tensor,
        # pegamos a média dos pesos no batch.
        if is_tensor_mode:
            final_weights_scalar = {loss: float(torch.mean(w)) for loss, w in final_weights.items()}
        else:
            final_weights_scalar = final_weights  # Já são floats

        # Retorna na ordem desejada
        return final_weights_scalar["mse"], final_weights_scalar["mae"], final_weights_scalar["log_cosh"]

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
    def __init__(self, model: torch.nn.Module, param_collection: "NamedParameterGroupCollection"):
        if param_collection is None:
            raise ValueError("param_collection não pode ser None para DeltaPatternRegularizer.")
        self.model = model
        self.param_collection: "NamedParameterGroupCollection" = param_collection
        self.initial_weights_run1: Dict[str, torch.Tensor] = {}  # Pesos no início da Run 1 (para salvar) - Em CPU
        self.initial_weights_run2: Dict[str, torch.Tensor] = (
            {}
        )  # Pesos no início da Run 2 (para calcular penalidade) - Em CPU
        self.reference_deltas: Dict[str, torch.Tensor] = {}  # Deltas carregados da Run 1 - Em CPU
        self.current_total_delta_norm: Optional[float] = None  # Cache da norma L2 do delta atual
        self.reference_delta_norm: Optional[float] = None  # Cache da norma L2 do delta de referência

    def _iterate_params(self) -> Iterable[Tuple[str, torch.Tensor, torch.device]]:
        """
        Itera diretamente sobre o state_dict do unet_lora e retorna todos os pesos reais com nome completo.
        NÃO depende de .named_parameters() para evitar perder buffers como alpha/dora_scale.
        """

        # INÍCIO ALTERAÇÃO: Cache do target_device na primeira iteração
        first_param = True
        # FIM ALTERAÇÃO

        for name, tensor in self.model.unet_lora.state_dict().items():
            if not isinstance(tensor, torch.Tensor):
                continue
            # INÍCIO ALTERAÇÃO: Cache do target_device
            if first_param:
                self.target_device = tensor.device
                first_param = False
            # FIM ALTERAÇÃO
            yield name, tensor, tensor.device

    def capture_weights(self):
        """Captura os pesos iniciais (usado no início da Run 1) e armazena em CPU."""
        self.initial_weights_run1 = {}  # Limpa antes de capturar
        count = 0
        try:
            for key, param, _ in self._iterate_params():
                self.initial_weights_run1[key] = param.to("cpu")
                count += 1
        except Exception as e:
            print(f"[DeltaPattern] Erro durante capture_weights: {e}")
            traceback.print_exc()
            raise  # Re-levanta a exceção para indicar falha

        if count == 0:
            warnings.warn("[DeltaPattern] capture_weights não encontrou parâmetros treináveis.", UserWarning)
        else:
            print(f"[DeltaPattern] Capturou pesos iniciais (Run 1) para {count} parâmetros treináveis.")
            # Opcional: Calcular e logar norma inicial aqui se desejado
            # initial_norm = self._calculate_total_norm(self.initial_weights_run1)
            # print(f"[DeltaPattern] Norma L2 total dos pesos iniciais (Run 1): {initial_norm:.4f}")

    def capture_initial_weights_run2(self):
        """
        Captura os pesos iniciais da Run 2 (usado para cálculo da penalidade).
        Armazena em CPU (`initial_weights_run2`) e também move uma cópia para
        o device do modelo (`initial_weights_run2_device`) para cálculo eficiente da penalidade.
        """
        # INÍCIO ALTERAÇÃO: Cachear também na GPU
        self.initial_weights_run2 = {}  # Limpa antes de capturar
        count = 0
        try:
            for key, param, _ in self._iterate_params():
                self.initial_weights_run2[key] = param.to("cpu")  # Armazena em CPU
                count += 1
        except Exception as e:
            print(f"[DeltaPattern] Erro durante capture_initial_weights_run2: {e}")
            traceback.print_exc()
            raise

        # Cachear na GPU
        if count > 0:
            if self.target_device is None:
                # Tenta obter o device se ainda não foi definido
                try:
                    _, _, self.target_device = next(self._iterate_params())
                except StopIteration:
                    warnings.warn(
                        "[DeltaPattern] Não foi possível determinar o device do modelo para cachear pesos iniciais da Run 2 na GPU.",
                        UserWarning,
                    )
            """
            essa modificações foi suprimida pra economizar vram, tava causando OOM
            """
            # if self.target_device:
            #     self.initial_weights_run2_device = {
            #         k: v.to(self.target_device) for k, v in self.initial_weights_run2.items()
            #     }
            #     print(f"[DeltaPattern] Pesos iniciais (Run 2) cacheados também em {self.target_device}.")
            # else:
            #     self.initial_weights_run2_device = None  # Não foi possível cachear na GPU
            self.initial_weights_run2_device = None

        if count == 0:
            warnings.warn(
                "[DeltaPattern] capture_initial_weights_run2 não encontrou parâmetros treináveis.", UserWarning
            )
        else:
            print(f"[DeltaPattern] Capturou pesos iniciais (Run 2) para {count} parâmetros treináveis.")

    def load_reference_pattern(self, pattern_path: str):
        """Carrega o padrão de delta de referência de um arquivo para a CPU e para a GPU."""
        self.reference_delta_norm = None
        if not os.path.exists(pattern_path):
            warnings.warn(f"Arquivo de padrão de delta de referência não encontrado: {pattern_path}", UserWarning)
            return

        try:
            # Garante que carregue na CPU
            # INÍCIO ALTERAÇÃO: Carregar e cachear na GPU
            loaded_data = load_file(pattern_path, device="cpu")  # Força CPU primeiro

            if isinstance(loaded_data, dict):
                # Validação simples: verifica se as chaves parecem ser strings e valores são tensores
                if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in loaded_data.items()):
                    self.reference_deltas = loaded_data
                    print(
                        f"[DeltaPattern] Padrão de delta de referência carregado com {len(self.reference_deltas)} deltas de parâmetros (CPU)."
                    )

                    # Calcula a norma L2 total (CPU)
                    self.reference_delta_norm = self._calculate_total_norm(self.reference_deltas)
                    print(
                        f"[DeltaPattern] Norma L2 total do delta de referência (CPU): {self.reference_delta_norm:.4f}"
                    )

                    # Cachear na GPU também
                    if self.target_device is None:
                        try:
                            _, _, self.target_device = next(self._iterate_params())
                        except StopIteration:
                            warnings.warn(
                                "[DeltaPattern] Não foi possível determinar o device do modelo para cachear delta de referência na GPU.",
                                UserWarning,
                            )

                    if self.target_device:
                        self.reference_deltas_device = {
                            k: v.to(self.target_device) for k, v in self.reference_deltas.items()
                        }
                        print(f"[DeltaPattern] Delta de referência cacheados também em {self.target_device}.")
                else:
                    self.reference_deltas_device = None
                    warnings.warn(
                        f"Arquivo de padrão de delta carregado tem formato de chave/valor inesperado: {pattern_path}",
                    )
            else:
                warnings.warn(f"Arquivo de padrão de delta carregado não é um dicionário: {pattern_path}", UserWarning)
        except Exception as e:
            warnings.warn(f"Falha ao carregar padrão de delta de referência de {pattern_path}: {e}", UserWarning)
            self.reference_deltas = {}  # Limpa em caso de erro
            self.reference_deltas_device = None

    def _calculate_total_norm(
        self, weight_dict: Dict[str, Tensor], target_device: Optional[torch.device] = None
    ) -> float:
        """
        Calcula a norma L2 total sobre todos os tensores no dicionário.
        Se target_device for None, assume que os tensores estão na CPU.
        Se target_device for fornecido, move os tensores para lá antes de calcular."""
        if not weight_dict:
            return 0.0
        total_norm_sq = torch.tensor(0.0, dtype=torch.float64)  # Usa float64 para precisão na soma
        for tensor in weight_dict.values():
            # Garante que o tensor está em CPU e calcula a norma quadrada
            total_norm_sq += torch.norm(tensor.to("cpu"), p=2).pow(2).double()
        return torch.sqrt(total_norm_sq).item()

    def _calculate_total_norm_device(self, weight_dict: Dict[str, Tensor]) -> torch.Tensor:
        dtype = next(iter(weight_dict.values())).dtype  # Detecta dinamicamente (bf16 ou float32)
        """Calcula a norma L2 total sobre todos os tensores no dicionário (assume que estão no mesmo device). Retorna tensor escalar no device."""
        if not weight_dict:
            return torch.tensor(0.0, device=self.target_device)  # Usa device cacheado

        # Calcula normas quadradas no device e soma
        norm_sq_list = [torch.norm(tensor.to(dtype=dtype), p=2).pow(2) for tensor in weight_dict.values()]
        if not norm_sq_list:  # Caso weight_dict tenha chaves mas valores vazios/inválidos
            return torch.tensor(0.0, device=self.target_device)

        total_norm_sq = torch.stack(norm_sq_list).sum()
        return torch.sqrt(total_norm_sq)

    def compute_penalty(self, lambda_weight: float) -> torch.Tensor:
        """
        Calcula a penalidade MSE entre o delta acumulado atual (current - initial_run2)
        e o delta de referência (carregado da Run 1).
        Utiliza tensores cacheados na GPU para eficiência.
        Retorna um tensor escalar no device do modelo.
        """
        # INÍCIO ALTERAÇÃO: Usar tensores cacheados na GPU e acumular na GPU
        if self.target_device is None:
            target_device = torch.device("cpu")
            warnings.warn(
                "[DeltaPattern] Não foi possível determinar o device do modelo para retornar a penalidade. Usando CPU.",
                UserWarning,
            )
        else:
            target_device = self.target_device

        if not self.reference_deltas_device:
            return torch.tensor(0.0, device=target_device)

        total_penalty_device = torch.tensor(0.0, dtype=torch.float32, device=target_device)  # Acumula MSE na GPU
        current_deltas_device = {}  # Guarda deltas atuais (na GPU) para cálculo da norma L2
        num_params_matched = 0

        try:
            for key, current_param, device in self._iterate_params():
                if device != target_device:
                    warnings.warn(
                        f"[DeltaPattern] Parâmetro '{key}' está em device inesperado ({device}). Esperado: {target_device}. Pulando.",
                        UserWarning,
                    )
                    continue

                # Verifica se temos os dados necessários (de forma segura)
                if key in self.reference_deltas_device and (
                    (self.initial_weights_run2_device and key in self.initial_weights_run2_device)
                    or key in self.initial_weights_run2
                ):
                    if self.initial_weights_run2_device:
                        initial_weight_run2_gpu = self.initial_weights_run2_device[key]
                    else:
                        initial_weight_run2_gpu = self.initial_weights_run2[key].to(target_device)

                    ref_delta_gpu = self.reference_deltas_device[key]

                    current_delta = current_param - initial_weight_run2_gpu

                    dtype = torch.bfloat16 if torch.is_bf16_supported() else torch.float32
                    penalty_term = torch.nn.functional.mse_loss(
                        current_delta.to(dtype=dtype),
                        ref_delta_gpu.to(dtype=dtype),
                    )

                    total_penalty_device += penalty_term
                    current_deltas_device[key] = current_delta
                    num_params_matched += 1


        except Exception as e:
            print(f"[DeltaPattern] Erro durante compute_penalty ao iterar parâmetros: {e}")
            traceback.print_exc()
            self.current_total_delta_norm = None  # Invalida norma cacheada
            return torch.tensor(0.0, device=target_device)  # Retorna 0 em caso de erro

        if num_params_matched > 0:
            # Calcula a norma L2 total do delta *atual* usando os deltas na GPU
            # Guarda o valor escalar .item() no cache, mas o cálculo é na GPU
            current_delta_norm_tensor = self._calculate_total_norm_device(current_deltas_device)
            self.current_total_delta_norm = current_delta_norm_tensor.item()

            # Calcula a penalidade média pelos parâmetros comparados
            average_penalty_device = total_penalty_device / num_params_matched
        else:
            self.current_total_delta_norm = 0.0
            # Avisa se não houve correspondência, mas tínhamos os dados
            if self.reference_deltas_device and self.initial_weights_run2_device:
                warnings.warn(
                    "[DeltaPattern] Nenhum parâmetro correspondente encontrado entre o modelo atual e os dados cacheados (iniciais Run 2 / ref delta).",
                    UserWarning,
                )
            average_penalty_device = torch.tensor(0.0, device=target_device)  # Média é zero na GPU

        # Retorna a penalidade média ponderada, já no device alvo
        final_penalty = lambda_weight * average_penalty_device
        return final_penalty

    def get_delta_norms(self) -> Tuple[Optional[float], Optional[float]]:
        """Retorna a norma L2 total cacheada do delta atual e do delta de referência."""
        return self.current_total_delta_norm, self.reference_delta_norm  # Retorna floats cacheados

    def save_pattern(
        self, save_dir: str = "./training_pattern", base_filename: Optional[str] = None, filename_suffix: str = ""
    ):
        """
        Calcula o delta final da Run 1 (final_weights - initial_weights_run1),
        salva em um arquivo .pt e escreve um log .txt com metadados e normas.
        Permite especificar um nome base e um sufixo para o arquivo.
        Retorna o caminho do arquivo .pt salvo ou None em caso de falha.
        """
        if not self.initial_weights_run1:
            print(
                "[DeltaPattern] Erro: Pesos iniciais (Run 1) não foram capturados. Não é possível salvar o padrão delta."
            )
            return None

        print(f"[DeltaPattern] Calculando padrão de delta final (Run 1)...")
        final_deltas_cpu: Dict[str, torch.Tensor] = {}
        final_weights_list_cpu = {}  # Para calcular norma final
        num_params_processed = 0

        try:
            # Itera sobre os parâmetros atuais do modelo para obter os pesos finais
            for key, final_param, _ in self._iterate_params():
                # Verifica se temos o peso inicial correspondente da Run 1 (que está na CPU)
                if key in self.initial_weights_run1:
                    initial_weight_cpu = self.initial_weights_run1[key]
                    # Calcula o delta final na CPU: final (CPU) - inicial (CPU)
                    final_delta_cpu = final_param.to("cpu") - initial_weight_cpu
                    final_deltas_cpu[key] = final_delta_cpu
                    # Guarda o peso final na CPU também para calcular a norma final
                    final_weights_list_cpu[key] = final_param.to("cpu")
                    num_params_processed += 1
        except Exception as e:
            print(f"[DeltaPattern] Erro ao calcular deltas finais: {e}")
            traceback.print_exc()
            return None

        if num_params_processed == 0:
            print("[DeltaPattern] Erro: Nenhum parâmetro correspondente encontrado para calcular o delta final.")
            return None

        # Calcula as normas L2 totais usando os dicionários na CPU
        # _calculate_total_norm por padrão opera na CPU
        initial_weights_norm = self._calculate_total_norm(self.initial_weights_run1)
        final_weights_norm = self._calculate_total_norm(final_weights_list_cpu)
        final_delta_norm = self._calculate_total_norm(final_deltas_cpu)

        print(f"[DeltaPattern] Norma L2 Inicial (Run 1): {initial_weights_norm:.4f}")
        print(f"[DeltaPattern] Norma L2 Final (Run 1): {final_weights_norm:.4f}")
        print(f"[DeltaPattern] Norma L2 Delta Final (Run 1): {final_delta_norm:.4f}")

        # Cria diretório e nome de arquivo
        os.makedirs(save_dir, exist_ok=True)

        # Define o nome base do arquivo
        if base_filename:
            # Remove extensão .pt se presente no nome base vindo do config
            if base_filename.lower().endswith(".pt"):
                base_filename = base_filename[:-3]
        else:
            # Usa um nome genérico se não fornecido
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_filename = f"pattern_{timestamp}"  # Nome antigo como fallback

        # Adiciona o sufixo (pode conter epoch/step)
        full_base_filename = base_filename + filename_suffix

        pattern_path = os.path.join(save_dir, full_base_filename + ".safetensors")  # Alterado para safetensors
        log_path = os.path.join(save_dir, full_base_filename + ".txt")

        try:
            # Salva usando safetensors
            from safetensors.torch import save_file

            save_file(final_deltas_cpu, pattern_path)
            print(f"[DeltaPattern] Padrão de delta final salvo em {pattern_path}")

            with open(log_path, "w") as f:
                f.write(f"Padrão Delta Salvo: {pattern_path}\n")
                # Incluir informações sobre o passo/época se disponíveis no sufixo?
                # f.write(f"Sufixo (Epoch/Step info): {filename_suffix}\n")
                f.write(f"Timestamp da Criação do Arquivo: {time.strftime('%Y%m%d_%H%M%S')}\n")
                f.write(f"Número de deltas de parâmetros no padrão: {len(final_deltas_cpu)}\n")
                try:
                    # Tenta acessar o atributo interno ou iterar sobre a coleção
                    groups = getattr(self.param_collection, "_NamedParameterGroupCollection__groups", None)
                    if groups is None:
                        num_groups = len(list(self.param_collection))  # Tenta iterar se o atributo não existe
                    else:
                        num_groups = len(groups)
                    f.write(f"Número de grupos de parâmetros considerados: {num_groups}\n")
                except Exception as group_err:
                    print(
                        f"[DeltaPattern] Aviso: Não foi possível determinar o número de grupos de parâmetros: {group_err}"
                    )
                    f.write("Número de grupos de parâmetros considerados: (Não foi possível determinar)\n")
                f.write(f"Norma L2 Total dos Pesos Iniciais (Run 1): {initial_weights_norm:.4f}\n")
                f.write(f"Norma L2 Total dos Pesos Finais (capturados neste save): {final_weights_norm:.4f}\n")
                f.write(f"Norma L2 Total do Delta (Finais - Iniciais): {final_delta_norm:.4f}\n\n")
                f.write("Detalhes do Delta por Parâmetro (Top 20 por Norma L2):\n")

                sorted_deltas = sorted(
                    final_deltas_cpu.items(),
                    key=lambda item: torch.norm(item[1], p=2).item(),
                    reverse=True,
                )
                for name, tensor in sorted_deltas[:20]:
                    f.write(
                        f"- {name}: {torch.norm(tensor, p=2).item():.6f}\n"
                    )  # Calcula norma em float32
                if len(sorted_deltas) > 20:
                    f.write("...\n")

            del final_deltas_cpu
            del final_weights_list_cpu
            torch.cuda.empty_cache()
            return pattern_path
        except ImportError:
            print(
                "[DeltaPattern] Erro: Biblioteca 'safetensors' não encontrada. Não é possível salvar o padrão delta. Instale com 'pip install safetensors'."
            )
            return None
        except Exception as e:
            print(f"[DeltaPattern] Erro ao salvar o padrão delta ou log: {e}")
            traceback.print_exc()
            # Tenta remover arquivos parciais
            if os.path.exists(pattern_path):
                try:
                    os.remove(pattern_path)
                except OSError:
                    pass
            if os.path.exists(log_path):
                try:
                    os.remove(log_path)
                except OSError:
                    pass
            return None
