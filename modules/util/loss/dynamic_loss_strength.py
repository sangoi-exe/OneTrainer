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
        self.mse_losses.append(mse_loss.detach())  # .detach() só para segurança
        self.mae_losses.append(mae_loss.detach())
        self.log_cosh_losses.append(log_cosh_loss.detach())

    def compute_stats(self, values_tensor: Tensor):
        """
        Se self.use_mad=False -> retorna (mean, std)
        Se self.use_mad=True  -> retorna (median, MAD)
        """
        # Se você quiser tratar cada 'batch element' separadamente, pode concatenar
        # a deque num grande tensor. Ex.: torch.cat(list_of_tensors).
        # Abaixo assumimos que cada loss na deque seja 0-D (scalar) ou shape [1].
        # Caso contrário, você pode querer concatenar e flatten:
        #
        # values_tensor = torch.cat(list(self.mse_losses)).view(-1)
        #
        # e assim por diante.

        # Verifica se há pelo menos 2 itens
        if values_tensor.numel() < 2:

            return torch.tensor(0.0, device=values_tensor.device), torch.tensor(
                1e-8, device=values_tensor.device
            )  # Evita problemas no início do treino

        if not self.use_mad:
            mean_val = values_tensor.mean()
            std_val = values_tensor.std(unbiased=False)
            std_val = torch.clamp(std_val, min=1e-8)  # evita zero
            return mean_val, std_val
        else:
            median_val = values_tensor.median()
            abs_dev = torch.abs(values_tensor - median_val)
            mad_val = abs_dev.median()
            mad_val = torch.clamp(mad_val, min=1e-8)  # evita zero
            return median_val, mad_val

    def compute_z_scores(self, mse_loss: Tensor, mae_loss: Tensor, log_cosh_loss: Tensor):
        """
        Retorna (mse_z, mae_z, log_cosh_z) usando media+std ou mediana+MAD,
        dependendo de self.use_mad.
        """
        # Aqui, a deque armazena vários tensores (cada update).
        # Precisamos converter numa única lista/tensor para computar estatísticas.
        # Se cada item já for 0-D, podemos empilhar (stack). Se for shape [batch], concat.
        mse_hist = torch.stack(list(self.mse_losses)).to(mse_loss.device)
        mae_hist = torch.stack(list(self.mae_losses)).to(mae_loss.device)
        log_hist = torch.stack(list(self.log_cosh_losses)).to(log_cosh_loss.device)

        mse_center, mse_scale = self.compute_stats(mse_hist)
        mae_center, mae_scale = self.compute_stats(mae_hist)
        log_cosh_center, log_cosh_scale = self.compute_stats(log_hist)

        mse_z = (mse_loss - mse_center) / mse_scale
        mae_z = (mae_loss - mae_center) / mae_scale
        log_cosh_z = (log_cosh_loss - log_cosh_center) / log_cosh_scale

        return (mse_z, mae_z, log_cosh_z)


class DynamicLossStrength:
    def __init__(
        self,
        use_ema=False,
        ema_decay=0.9,
        outlier_threshold=10.0,
    ):
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.outlier_threshold = outlier_threshold

        # Estados para EMA
        self.mse_weight_ema = 1.0
        self.mae_weight_ema = 1.0
        self.log_cosh_weight_ema = 1.0
        self.initialized = False

    def adjust_weights(
        self,
        mse_z: torch.Tensor,
        mae_z: torch.Tensor,
        log_cosh_z: torch.Tensor,
        config,
        progress,
        epoch_length,
        loss_tracker: LossTracker = None,
    ):
        """
        Retorna (mse_weight, mae_weight, log_cosh_weight).
        Aplica:
        1) Clamping do z-score em outlier_threshold (elementwise se tiver shape)
        2) Inverse z-scores
        3) Normalização
        4) (Opcional) EMA
        5) Scheduler (prioridade ao MAE no início e MSE no final)
        """

        # 1) Clamping elementwise (se mse_z for multi-element)
        # Se for apenas 1 valor (0-D), clamp() ainda funciona do mesmo jeito.
        z_mse = torch.clamp(mse_z.abs(), max=self.outlier_threshold)
        z_mae = torch.clamp(mae_z.abs(), max=self.outlier_threshold)
        z_log = torch.clamp(log_cosh_z.abs(), max=self.outlier_threshold)

        # Se z_mse for um escalar 0-D, 'z_mse + z_mae + z_log' também será 0-D.
        total_z = z_mse + z_mae + z_log
        total_z = torch.clamp(total_z, min=1e-8)

        # 2) inverse z-scores
        inv_mse = (total_z - z_mse) / total_z
        inv_mae = (total_z - z_mae) / total_z
        inv_log = (total_z - z_log) / total_z

        # soma dos inversos
        sum_inv = inv_mse + inv_mae + inv_log
        sum_inv = torch.clamp(sum_inv, min=1e-8)

        # pesos normalizados
        mse_weight = inv_mse / sum_inv
        mae_weight = inv_mae / sum_inv
        log_cosh_weight = inv_log / sum_inv

        # 3) EMA (se habilitado)
        if self.use_ema:
            # Precisamos converter para float se for 0-D
            mse_val = mse_weight.item() if mse_weight.numel() == 1 else float(mse_weight.mean())
            mae_val = mae_weight.item() if mae_weight.numel() == 1 else float(mae_weight.mean())
            log_val = log_cosh_weight.item() if log_cosh_weight.numel() == 1 else float(log_cosh_weight.mean())

            if not self.initialized:
                self.mse_weight_ema = mse_val
                self.mae_weight_ema = mae_val
                self.log_cosh_weight_ema = log_val
                self.initialized = True
            else:
                alpha = 1.0 - self.ema_decay
                self.mse_weight_ema = (1 - alpha) * self.mse_weight_ema + alpha * mse_val
                self.mae_weight_ema = (1 - alpha) * self.mae_weight_ema + alpha * mae_val
                self.log_cosh_weight_ema = (1 - alpha) * self.log_cosh_weight_ema + alpha * log_val

            # Normaliza a soma do EMA
            sum_ema = self.mse_weight_ema + self.mae_weight_ema + self.log_cosh_weight_ema
            if sum_ema < 1e-8:
                sum_ema = 1.0

            self.mse_weight_ema /= sum_ema
            self.mae_weight_ema /= sum_ema
            self.log_cosh_weight_ema /= sum_ema

            # Retorna como tensores 0-D
            mse_weight = torch.tensor(self.mse_weight_ema, device=mse_weight.device)
            mae_weight = torch.tensor(self.mae_weight_ema, device=mae_weight.device)
            log_cosh_weight = torch.tensor(self.log_cosh_weight_ema, device=log_cosh_weight.device)

        # 4) Scheduler de prioridade ao longo das épocas/steps
        total_steps = config.epochs * epoch_length
        if total_steps > 0:
            frac = (progress.epoch * epoch_length + progress.epoch_step) / float(total_steps)
            frac = max(0.0, min(frac, 1.0))
        else:
            frac = 0.0

        # if frac < 0.3:
        #     # até 30% do treino
        #     self.use_ema = True
        #     if loss_tracker is not None:
        #         loss_tracker.use_mad = True
        # else:
        #     # depois de 70% do treino
        #     self.use_ema = False
        #     if loss_tracker is not None:
        #         loss_tracker.use_mad = False

        if frac < 0.20:
            t = frac / 0.20  # Normaliza para o intervalo [0, 1]
            mae_sched = 0.5 * (1 - t)
            log_sched = 0.5 + 0.5 * t
            mse_sched = 0.0
        elif frac < 0.40:
            t = (frac - 0.20) / 0.20
            mae_sched = 0.0
            log_sched = 1.0 - 0.5 * t
            mse_sched = 0.5 * t
        elif frac < 0.75:
            t = (frac - 0.40) / 0.35
            mae_sched = 0.0
            log_sched = 0.5 * (1 - t)
            mse_sched = 0.5 + 0.5 * t
        else:
            t = min(max((frac - 0.75) / 0.25, 0.0), 1.0)
            mae_sched = 0.0
            log_sched = 0.5 * (1 - t)
            mse_sched = 0.5 + 0.5 * t


        # Multiplicar
        # Se forem 0-D, ok. Se forem multi-element, multiplicação elementwise
        mse_weight = mse_weight * mse_sched
        mae_weight = mae_weight * mae_sched
        log_cosh_weight = log_cosh_weight * log_sched

        # Renormaliza (opcional)
        sum_w = mse_weight + mae_weight + log_cosh_weight
        sum_w = torch.clamp(sum_w, min=1e-8)

        mse_weight = mse_weight / sum_w
        mae_weight = mae_weight / sum_w
        log_cosh_weight = log_cosh_weight / sum_w

        return mse_weight, mae_weight, log_cosh_weight
