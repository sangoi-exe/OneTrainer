import torch
from torch import Tensor
from collections import deque
from typing import Tuple, List, Dict, Union

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
        self,
        mse_loss: Tensor,
        mae_loss: Tensor,
        log_cosh_loss: Tensor
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

        # Initialize scheduling parameters
        self.schedule_params: Dict[str, Dict[str, float]] = self._initialize_schedule_params(schedule_params)

        # EMA state
        self.ema_weights: Dict[str, float] = {"mse": 1.0, "mae": 1.0, "log_cosh": 1.0}
        self.initialized: bool = False

    def _initialize_schedule_params(self, schedule_params: Dict[str, Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
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
                    self.ema_weights[loss] = \
                        (1 - alpha) * self.ema_weights[loss] + alpha * base_weights[loss]

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
        weighted_weights = {
            loss: base_weights[loss] * scheduled_factors[loss]
            for loss in base_weights
        }

        # Normaliza
        sum_weighted = sum(weighted_weights.values())
        if sum_weighted < 1e-8:
            sum_weighted = 1.0

        final_weights = {loss: w / sum_weighted for loss, w in weighted_weights.items()}

        # Retorna na ordem desejada
        return final_weights["mse"], final_weights["mae"], final_weights["log_cosh"]
