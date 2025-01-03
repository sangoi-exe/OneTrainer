import torch
from torch import Tensor
from collections import deque


class LossTracker:
    """
    Armazena e gera 'estatísticas' das últimas N perdas para MSE, MAE e log-cosh.
    Pode usar média/desvio ou mediana/MAD.
    """

    def __init__(self, window_size=100, use_mad=False):
        self.window_size = window_size
        self.use_mad = use_mad

        self.mse_losses = deque(maxlen=window_size)
        self.mae_losses = deque(maxlen=window_size)
        self.log_cosh_losses = deque(maxlen=window_size)

    def update(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor):
        # Armazena o item() de cada perda
        self.mse_losses.append(mse_loss.item())
        self.mae_losses.append(mae_loss.item())
        self.log_cosh_losses.append(log_cosh_loss.item())

    def compute_stats(self, values_list):
        """
        Se use_mad=False -> retorna (mean, std)
        Se use_mad=True -> retorna (median, MAD)
        """
        if len(values_list) < 2:
            # Evita problemas no início do treino
            return 0.0, 1e-8

        arr = torch.tensor(values_list, dtype=torch.float32)
        if not self.use_mad:
            mean_val = arr.mean()
            std_val = arr.std(unbiased=False)
            return mean_val.item(), max(std_val.item(), 1e-8)
        else:
            median_val = arr.median()
            abs_dev = torch.abs(arr - median_val)
            mad_val = abs_dev.median()
            return median_val.item(), max(mad_val.item(), 1e-8)

    def compute_z_scores(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor):
        """
        Retorna (mse_z, mae_z, log_cosh_z) usando media+std ou mediana+MAD,
        dependendo de self.use_mad.
        """
        mse_center, mse_scale = self.compute_stats(self.mse_losses)
        mae_center, mae_scale = self.compute_stats(self.mae_losses)
        log_cosh_center, log_cosh_scale = self.compute_stats(self.log_cosh_losses)

        mse_z = (mse_loss.item() - mse_center) / mse_scale
        mae_z = (mae_loss.item() - mae_center) / mae_scale
        log_cosh_z = (log_cosh_loss.item() - log_cosh_center) / log_cosh_scale

        return (mse_z, mae_z, log_cosh_z)


class DynamicLossStrength:
    def __init__(
        self,
        use_ema=False,
        ema_decay=0.9,
        outlier_threshold=3.0,
        # Caso queira já passar valores default para a schedule
        mae_start=0.6,
        mae_end=0.2,
        mse_start=0.2,
        mse_end=0.6,
        log_start=0.2,
        log_end=0.2,
    ):
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.outlier_threshold = outlier_threshold

        # Guardar parâmetros de schedule
        self.mae_start = mae_start
        self.mae_end = mae_end
        self.mse_start = mse_start
        self.mse_end = mse_end
        self.log_start = log_start
        self.log_end = log_end

        # Estados para EMA
        self.mse_weight_ema = 1.0
        self.mae_weight_ema = 1.0
        self.log_cosh_weight_ema = 1.0
        self.initialized = False

    def adjust_weights(self, mse_z: float, mae_z: float, log_cosh_z: float, config, progress):
        """
        Retorna (mse_weight, mae_weight, log_cosh_weight).
        Aplica:
        1) Clamping do z-score em outlier_threshold
        2) Inverse z-scores
        3) Normalização
        4) (Opcional) EMA
        5) Scheduler (prioridade ao MAE no início e MSE no final)
        """

        # 1) Clamping
        z_mse = min(abs(mse_z), self.outlier_threshold)
        z_mae = min(abs(mae_z), self.outlier_threshold)
        z_log = min(abs(log_cosh_z), self.outlier_threshold)

        total_z = z_mse + z_mae + z_log
        if total_z < 1e-8:
            total_z = 1.0

        # 2) inverse z-scores
        inv_mse = (total_z - z_mse) / total_z
        inv_mae = (total_z - z_mae) / total_z
        inv_log = (total_z - z_log) / total_z

        sum_inv = inv_mse + inv_mae + inv_log
        if sum_inv < 1e-8:
            sum_inv = 1.0

        # pesos normalizados baseados em z-score
        mse_weight = inv_mse / sum_inv
        mae_weight = inv_mae / sum_inv
        log_cosh_weight = inv_log / sum_inv

        # 3) EMA (se habilitado)
        if self.use_ema:
            if not self.initialized:
                # primeira chamada
                self.mse_weight_ema = mse_weight
                self.mae_weight_ema = mae_weight
                self.log_cosh_weight_ema = log_cosh_weight
                self.initialized = True
            else:
                alpha = 1.0 - self.ema_decay
                self.mse_weight_ema = (1 - alpha) * self.mse_weight_ema + alpha * mse_weight
                self.mae_weight_ema = (1 - alpha) * self.mae_weight_ema + alpha * mae_weight
                self.log_cosh_weight_ema = (1 - alpha) * self.log_cosh_weight_ema + alpha * log_cosh_weight

            # Normaliza a soma do EMA
            sum_ema = self.mse_weight_ema + self.mae_weight_ema + self.log_cosh_weight_ema
            if sum_ema < 1e-8:
                sum_ema = 1.0
            self.mse_weight_ema /= sum_ema
            self.mae_weight_ema /= sum_ema
            self.log_cosh_weight_ema /= sum_ema

            mse_weight = self.mse_weight_ema
            mae_weight = self.mae_weight_ema
            log_cosh_weight = self.log_cosh_weight_ema

        # 4) Scheduler de prioridade ao longo das épocas
        # fração do treino (0.0 no começo, 1.0 no final)
        if config.epochs > 1:
            frac = progress.epoch / float(config.epochs - 1)
        else:
            frac = 0.0
        frac = max(0.0, min(frac, 1.0))  # clamp

        # Interpolar: MAE = 0.6 -> 0.2, MSE = 0.2 -> 0.6, log = 0.2 -> 0.2
        mae_sched = self.mae_start * (1 - frac) + self.mae_end * frac
        mse_sched = self.mse_start * (1 - frac) + self.mse_end * frac
        log_sched = self.log_start * (1 - frac) + self.log_end * frac

        # Multiplica os pesos dinâmicos pelos fatores de schedule
        mse_weight *= mse_sched
        mae_weight *= mae_sched
        log_cosh_weight *= log_sched

        # Renormaliza, se quiser manter a soma = 1
        sum_sch = mse_weight + mae_weight + log_cosh_weight
        if sum_sch < 1e-8:
            sum_sch = 1.0

        mse_weight /= sum_sch
        mae_weight /= sum_sch
        log_cosh_weight /= sum_sch

        # Retorna pesos finais
        return mse_weight, mae_weight, log_cosh_weight
