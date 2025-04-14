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

    def compute_stats(self, values_list: List[Union[float, Tensor]]) -> Tuple[float, float]:
        try:
            arr = torch.cat([x.view(-1) for x in values_list], dim=0)
            if not self.use_mad:
                mean_val = arr.mean()
                std_val = arr.std(unbiased=False)
                return float(mean_val), max(float(std_val), 1e-8)
            else:
                median_val = arr.median()
                abs_dev = torch.abs(arr - median_val)
                mad_val = abs_dev.median()
                return median_val.item(), max(mad_val.item(), 1e-8)
        except RuntimeError as oom:
            print(f"[LossTracker] OOM detected in compute_stats: fallback to mean-only. Err: {oom}")
            mean_vals = [x.mean().item() if isinstance(x, Tensor) else x for x in values_list]
            arr = torch.tensor(mean_vals)
            return arr.mean().item(), max(arr.std(unbiased=False).item(), 1e-8)

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
                self.initial_weights_run1[key] = param.detach().clone().cpu()  # Armazena em CPU
                count += 1
        except Exception as e:
            print(f"[DeltaPattern] Erro durante capture_weights: {e}")
            traceback.print_exc()
            raise  # Re-levanta a exceção para indicar falha

        if count == 0:
            print("[DeltaPattern] capture_weights não encontrou parâmetros treináveis.")
        else:
            print(f"[DeltaPattern] Capturou pesos iniciais (Run 1) para {count} parâmetros treináveis.")
            # Opcional: Calcular e logar norma inicial aqui se desejado
            # initial_norm = self._calculate_total_norm(self.initial_weights_run1)
            # print(f"[DeltaPattern] Norma L2 total dos pesos iniciais (Run 1): {initial_norm:.4f}")

    def capture_initial_weights_run2(self):
        """Captura os pesos iniciais da Run 2 (usado para cálculo da penalidade) e armazena em CPU."""
        self.initial_weights_run2 = {}  # Limpa antes de capturar
        count = 0
        try:
            for key, param, _ in self._iterate_params():
                self.initial_weights_run2[key] = param.detach().clone().cpu()  # Armazena em CPU
                count += 1
        except Exception as e:
            print(f"[DeltaPattern] Erro durante capture_initial_weights_run2: {e}")
            traceback.print_exc()
            raise

        if count == 0:
            print("[DeltaPattern] capture_initial_weights_run2 não encontrou parâmetros treináveis.")
        else:
            print(f"[DeltaPattern] Capturou pesos iniciais (Run 2) para {count} parâmetros treináveis.")
            # Opcional: Calcular e logar norma inicial aqui se desejado
            # initial_norm_run2 = self._calculate_total_norm(self.initial_weights_run2)
            # print(f"[DeltaPattern] Norma L2 total dos pesos iniciais (Run 2): {initial_norm_run2:.4f}")

    def load_reference_pattern(self, pattern_path: str):
        """Carrega o padrão de delta de referência de um arquivo para a CPU."""
        self.reference_deltas = {}  # Reseta antes de carregar
        self.reference_delta_norm = None
        if not os.path.exists(pattern_path):
            print(f"Arquivo de padrão de delta de referência não encontrado: {pattern_path}")
            return

        try:
            # Garante que carregue na CPU
            loaded_data = load_file(pattern_path)
            if isinstance(loaded_data, dict):
                # Validação simples: verifica se as chaves parecem ser strings e valores são tensores
                if all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in loaded_data.items()):
                    self.reference_deltas = loaded_data
                    print(
                        f"[DeltaPattern] Padrão de delta de referência carregado com {len(self.reference_deltas)} deltas de parâmetros."
                    )
                    # Calcula a norma L2 total dos deltas carregados (já estão em CPU)
                    self.reference_delta_norm = self._calculate_total_norm(self.reference_deltas)
                    print(f"[DeltaPattern] Norma L2 total do delta de referência: {self.reference_delta_norm:.4f}")
                else:
                    print(
                        f"Arquivo de padrão de delta carregado tem formato de chave/valor inesperado: {pattern_path}",
                        UserWarning,
                    )
                    self.reference_deltas = {}  # Limpa se o formato estiver incorreto
            else:
                print(f"Arquivo de padrão de delta carregado não é um dicionário: {pattern_path}")
        except Exception as e:
            print(f"Falha ao carregar padrão de delta de referência de {pattern_path}: {e}")
            self.reference_deltas = {}  # Limpa em caso de erro

    def _calculate_total_norm(self, weight_dict: Dict[str, Tensor]) -> float:
        """Calcula a norma L2 total sobre todos os tensores no dicionário (assume que estão em CPU)."""
        if not weight_dict:
            return 0.0
        total_norm_sq = torch.tensor(0.0, dtype=torch.float64)  # Usa float64 para precisão na soma
        for tensor in weight_dict.values():
            # Garante que o tensor está em CPU e calcula a norma quadrada
            total_norm_sq += torch.norm(tensor.float().cpu(), p=2).pow(2).double()
        return torch.sqrt(total_norm_sq).item()

    def compute_penalty(self, lambda_weight: float) -> torch.Tensor:
        """
        Calcula a penalidade MSE entre o delta acumulado atual (current - initial_run2)
        e o delta de referência (carregado da Run 1).
        Retorna um tensor escalar no device do primeiro parâmetro do modelo.
        """
        # Tenta obter o device do primeiro parâmetro para o tensor de retorno (ou usa CPU como fallback)
        try:
            _, _, target_device = next(self._iterate_params())
        except StopIteration:
            target_device = torch.device("cpu")
            print(
                "[DeltaPattern] Não foi possível determinar o device do modelo para retornar a penalidade. Usando CPU.",
                UserWarning,
            )

        # Verifica se temos tudo necessário
        if not self.reference_deltas or not self.initial_weights_run2:
            if not self.reference_deltas:
                # Log apenas uma vez ou periodicamente para não poluir
                pass  # print("[DeltaPattern] Aviso: Padrão de delta de referência não carregado. Penalidade será 0.")
            if not self.initial_weights_run2:
                # Log apenas uma vez ou periodicamente
                pass  # print("[DeltaPattern] Aviso: Pesos iniciais da Run 2 não capturados. Penalidade será 0.")
            self.current_total_delta_norm = 0.0  # Reseta norma atual se não puder calcular
            return torch.tensor(0.0, device=target_device)

        total_penalty_cpu = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # Acumula MSE em CPU
        current_deltas_for_norm = {}  # Guarda deltas atuais (em CPU) para cálculo da norma L2
        num_params_matched = 0

        try:
            # INÍCIO ALTERAÇÃO: Cache dos tensores já no device
            target_device = next(self.model.parameters()).device
            initial_run2_device = {
                k: v.to(dtype=torch.float32, device=target_device)
                for k, v in self.initial_weights_run2.items()
            }
            ref_delta_device = {
                k: v.to(dtype=torch.float32, device=target_device)
                for k, v in self.reference_deltas.items()
            }
            # FIM ALTERAÇÃO

            for key, current_param, device in self._iterate_params():
                if key in ref_delta_device and key in initial_run2_device:
                    initial_weight_run2 = initial_run2_device[key]
                    ref_delta = ref_delta_device[key]

                    current_delta = current_param.detach().to(dtype=torch.float32) - initial_weight_run2

                    penalty_term = torch.nn.functional.mse_loss(current_delta, ref_delta)

                    # INÍCIO ALTERAÇÃO: evitar cópia para CPU desnecessária
                    total_penalty_cpu += penalty_term.to("cpu")
                    current_deltas_for_norm[key] = current_delta.to("cpu")
                    # FIM ALTERAÇÃO

                    num_params_matched += 1

        except Exception as e:
            print(f"[DeltaPattern] Erro durante compute_penalty ao iterar parâmetros: {e}")
            traceback.print_exc()
            self.current_total_delta_norm = None  # Invalida norma cacheada
            return torch.tensor(0.0, device=target_device)  # Retorna 0 em caso de erro

        if num_params_matched > 0:
            # Calcula a norma L2 total do delta *atual* usando os deltas em CPU
            self.current_total_delta_norm = self._calculate_total_norm(current_deltas_for_norm)
            # Calcula a penalidade média pelos parâmetros comparados
            average_penalty_cpu = total_penalty_cpu / num_params_matched
        else:
            self.current_total_delta_norm = 0.0
            # Avisa se não houve correspondência, mas tínhamos os dados
            if self.reference_deltas and self.initial_weights_run2:
                print(
                    "[DeltaPattern] Nenhum parâmetro correspondente encontrado entre o modelo atual, pesos iniciais da Run 2 e padrão de delta de referência.",
                    UserWarning,
                )
            average_penalty_cpu = torch.tensor(0.0, device="cpu")

        # Retorna a penalidade média ponderada, movida para o device alvo
        final_penalty = (lambda_weight * average_penalty_cpu).to(target_device)
        return final_penalty

    def get_delta_norms(self) -> Tuple[Optional[float], Optional[float]]:
        """Retorna a norma L2 total cacheada do delta atual e do delta de referência."""
        return self.current_total_delta_norm, self.reference_delta_norm

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
            for key, final_param, _ in self._iterate_params():
                if key in self.initial_weights_run1:
                    initial_weight_cpu = self.initial_weights_run1[key]
                    final_delta_cpu = final_param.detach().cpu() - initial_weight_cpu
                    final_deltas_cpu[key] = final_delta_cpu
                    final_weights_list_cpu[key] = final_param.detach().cpu()
                    num_params_processed += 1
        except Exception as e:
            print(f"[DeltaPattern] Erro ao calcular deltas finais: {e}")
            traceback.print_exc()
            return None

        if num_params_processed == 0:
            print("[DeltaPattern] Erro: Nenhum parâmetro correspondente encontrado para calcular o delta final.")
            return None

        # Calcula as normas L2 totais (todos os tensores estão em CPU)
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

        pattern_path = os.path.join(save_dir, full_base_filename + ".pt")
        log_path = os.path.join(save_dir, full_base_filename + ".txt")

        try:
            torch.save(final_deltas_cpu, pattern_path)
            print(f"[DeltaPattern] Padrão de delta final salvo em {pattern_path}")

            with open(log_path, "w") as f:
                f.write(f"Padrão Delta Salvo: {pattern_path}\n")
                # Incluir informações sobre o passo/época se disponíveis no sufixo?
                # f.write(f"Sufixo (Epoch/Step info): {filename_suffix}\n")
                f.write(f"Timestamp da Criação do Arquivo: {time.strftime('%Y%m%d_%H%M%S')}\n")
                f.write(f"Número de deltas de parâmetros no padrão: {len(final_deltas_cpu)}\n")
                try:
                    num_groups = len(
                        getattr(
                            self.param_collection, "_NamedParameterGroupCollection__groups", list(self.param_collection)
                        )
                    )
                    f.write(f"Número de grupos de parâmetros considerados: {num_groups}\n")
                except:
                    f.write("Número de grupos de parâmetros considerados: (Não foi possível determinar)\n")
                f.write(f"Norma L2 Total dos Pesos Iniciais (Run 1): {initial_weights_norm:.4f}\n")
                f.write(f"Norma L2 Total dos Pesos Finais (capturados neste save): {final_weights_norm:.4f}\n")
                f.write(f"Norma L2 Total do Delta (Finais - Iniciais): {final_delta_norm:.4f}\n\n")
                f.write("Detalhes do Delta por Parâmetro (Top 20 por Norma L2):\n")

                sorted_deltas = sorted(
                    final_deltas_cpu.items(), key=lambda item: torch.norm(item[1], p=2).item(), reverse=True
                )
                for name, tensor in sorted_deltas[:20]:
                    f.write(f"- {name}: {torch.norm(tensor, p=2).item():.6f}\n")
                if len(sorted_deltas) > 20:
                    f.write("...\n")

            return pattern_path
        except Exception as e:
            print(f"[DeltaPattern] Erro ao salvar o padrão delta ou log: {e}")
            traceback.print_exc()
            if os.path.exists(pattern_path):
                os.remove(pattern_path)
            if os.path.exists(log_path):
                os.remove(log_path)
            return None
