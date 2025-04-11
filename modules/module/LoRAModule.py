import re
import os
import math
import copy
import fnmatch
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Conv2d, Dropout, Linear, Parameter
from typing import Any
from abc import abstractmethod
from collections.abc import Mapping
from modules.util.enum.ModelType import PeftType
from modules.util.config.TrainConfig import TrainConfig
from modules.util.quantization_util import get_unquantized_weight, get_weight_shape


class RuleSet:
    def __init__(self, pattern_dict: dict[str, dict[str, int | float]]):
        self.patterns = pattern_dict or {}
        self._sorted_patterns = sorted(self.patterns.keys(), key=len, reverse=True)
        # print(f"[RuleSet DEBUG] Initialized with patterns: {self.patterns}")
        # print(f"[RuleSet DEBUG] Sorted patterns for matching: {self._sorted_patterns}")

    def match(self, name: str) -> dict[str, int | float] | None:
        """Encontra o primeiro padrão (do mais específico para o mais geral) que corresponde ao nome e retorna sua configuração."""
        for pattern in self._sorted_patterns:
            # Usar fnmatch diretamente é mais simples para padrões glob
            if fnmatch.fnmatch(name, pattern):
                # print(f"[RuleSet DEBUG] Match FOUND for '{name}' with pattern '{pattern}'") # Debug Match
                # Retorna o dicionário de configuração associado ao padrão encontrado
                return self.patterns[pattern]
        # print(f"[RuleSet DEBUG] No match found for '{name}'") # Debug No Match (Opcional, pode poluir)
        return None  # Retorna None se nenhum padrão corresponder


# --- Adicionado: Definição de blocos e mapeamento ---
BLOCKID26 = [
    "BASE",
    "IN00",
    "IN01",
    "IN02",
    "IN03",
    "IN04",
    "IN05",
    "IN06",
    "IN07",
    "IN08",
    "IN09",
    "IN10",
    "IN11",
    "M00",
    "OUT00",
    "OUT01",
    "OUT02",
    "OUT03",
    "OUT04",
    "OUT05",
    "OUT06",
    "OUT07",
    "OUT08",
    "OUT09",
    "OUT10",
    "OUT11",
]
KEY_TO_BLOCK_MAPPING = {
    "lora_unet_conv_in": "IN00",
    "lora_unet_time_embedding": "IN09",  # Agrupando embeddings aqui por convenção
    "lora_unet_add_embedding": "IN09",  # Agrupando embeddings aqui por convenção
    "lora_unet_down_blocks_0_resnets_0": "IN01",
    "lora_unet_down_blocks_0_resnets_1": "IN02",
    "lora_unet_down_blocks_0_downsamplers_0": "IN03",
    "lora_unet_down_blocks_1_resnets_0": "IN04",
    "lora_unet_down_blocks_1_attentions_0": "IN04",
    "lora_unet_down_blocks_1_resnets_1": "IN05",
    "lora_unet_down_blocks_1_attentions_1": "IN05",
    "lora_unet_down_blocks_1_downsamplers_0": "IN06",
    "lora_unet_down_blocks_2_resnets_0": "IN07",
    "lora_unet_down_blocks_2_attentions_0": "IN07",
    "lora_unet_down_blocks_2_resnets_1": "IN08",
    "lora_unet_down_blocks_2_attentions_1": "IN08",
    # IN09 já tratado pelos embeddings
    # IN10, IN11 tipicamente vazios
    "lora_unet_mid_block": "M00",  # Cobre resnets e attentions do mid block
    "lora_unet_up_blocks_0_resnets_0": "OUT00",
    "lora_unet_up_blocks_0_attentions_0": "OUT00",
    "lora_unet_up_blocks_0_resnets_1": "OUT01",
    "lora_unet_up_blocks_0_attentions_1": "OUT01",
    "lora_unet_up_blocks_0_resnets_2": "OUT02",
    "lora_unet_up_blocks_0_attentions_2": "OUT02",
    "lora_unet_up_blocks_0_upsamplers_0": "OUT02",
    "lora_unet_up_blocks_1_resnets_0": "OUT03",
    "lora_unet_up_blocks_1_attentions_0": "OUT03",
    "lora_unet_up_blocks_1_resnets_1": "OUT04",
    "lora_unet_up_blocks_1_attentions_1": "OUT04",
    "lora_unet_up_blocks_1_resnets_2": "OUT05",
    "lora_unet_up_blocks_1_attentions_2": "OUT05",
    "lora_unet_up_blocks_1_upsamplers_0": "OUT05",
    "lora_unet_up_blocks_2_resnets_0": "OUT06",
    "lora_unet_up_blocks_2_resnets_1": "OUT07",
    "lora_unet_up_blocks_2_resnets_2": "OUT08",
    # OUT09, OUT10 tipicamente vazios
    "lora_unet_conv_out": "OUT11",
}
# --- Fim da Adição ---


class PeftBase(nn.Module):
    is_applied: bool
    orig_forward: Any | None
    orig_eval: Any | None
    orig_train: Any | None
    _orig_module: list[nn.Module] | None  # list prevents it from registering
    prefix: str
    layer_kwargs: dict  # Applied during the forward op() call.
    _initialized: bool  # Tracks whether we've created the layers or not.

    def __init__(self, prefix: str, orig_module: nn.Module | None):
        super().__init__()
        # Garante que o prefixo seja seguro para nomes de arquivos/chaves
        self.prefix = prefix.replace(".", "_") + "."
        self._orig_module = [orig_module] if orig_module else None
        self.is_applied = False
        self.layer_kwargs = {}
        self._initialized = False

        if orig_module is not None:
            match orig_module:
                case nn.Linear():
                    self.op = F.linear
                    self.shape = get_weight_shape(orig_module)
                case nn.Conv2d():
                    self.op = F.conv2d
                    self.shape = get_weight_shape(orig_module)
                    self.layer_kwargs.setdefault("stride", orig_module.stride)
                    self.layer_kwargs.setdefault("padding", orig_module.padding)
                    self.layer_kwargs.setdefault("dilation", orig_module.dilation)
                    self.layer_kwargs.setdefault("groups", orig_module.groups)
                case _:
                    # Permite que a inicialização continue mesmo para tipos não suportados,
                    # mas não configura op, shape, etc. A lógica de criação falhará mais tarde.
                    print(
                        f"Aviso: Tipo de módulo não suportado encontrado em PeftBase: {type(orig_module)}. Continuando, mas pode falhar."
                    )
                    self.op = None
                    self.shape = None
                    # raise NotImplementedError("Only Linear and Conv2d are supported layers.")

    def hook_to_module(self):
        if self.orig_module is None:
            return  # Não pode fazer hook sem módulo original
        if not self.is_applied:
            self.orig_forward = self.orig_module.forward
            self.orig_train = self.orig_module.train
            self.orig_eval = self.orig_module.eval
            self.orig_module.forward = self.forward
            self.orig_module.train = self._wrap_train
            self.orig_module.eval = self._wrap_eval
            self.is_applied = True

    def remove_hook_from_module(self):
        if self.orig_module is None or self.orig_forward is None:
            return
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.orig_module.train = self.orig_train
            self.orig_module.eval = self.orig_eval
            self.is_applied = False

    def _wrap_train(self, mode=True):
        if self.orig_module is None or self.orig_train is None:
            return
        self.orig_train(mode)
        self.train(mode)

    def _wrap_eval(self):
        if self.orig_module is None or self.orig_eval is None:
            return
        self.orig_eval()
        self.eval()

    def make_weight(self, A: Tensor, B: Tensor):
        """Layer-type-independent way of creating a weight matrix from LoRA A/B."""
        if self.shape is None:
            raise RuntimeError(
                f"Cannot make weight for {self.prefix}: shape not determined (likely unsupported layer type)."
            )
        W = B.view(B.size(0), -1) @ A.view(A.size(0), -1)
        return W.view(self.shape)

    def check_initialized(self):
        """Checks, and raises an exception, if the module is not initialized."""
        if not self._initialized:
            raise RuntimeError(f"Module {self.prefix} is not initialized.")

        # Perform assertions to make pytype happy.
        assert self.orig_module is not None, f"orig_module is None for {self.prefix}"
        assert self.orig_forward is not None, f"orig_forward is None for {self.prefix}"
        # op pode ser None para tipos não suportados, mas forward deve lidar com isso
        # assert self.op is not None

    @property
    def orig_module(self) -> nn.Module:
        if self._orig_module is None:
            raise AttributeError(
                f"Original module not set for PEFT layer with prefix {self.prefix}"
            )
        return self._orig_module[0]

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # Filtra o state_dict para conter apenas chaves relevantes para este módulo específico
        module_prefix_with_dot = self.prefix  # self.prefix já termina com '.'
        relevant_keys = {
            k: v for k, v in state_dict.items() if k.startswith(module_prefix_with_dot)
        }
        if not relevant_keys:
            # Se nenhuma chave relevante for encontrada, talvez o módulo não esteja no state_dict
            # ou o prefixo está incorreto. Retorna chaves vazias para indicar isso.
            # Não inicializa o módulo se não houver chaves para carregar.
            # print(f"Warning: No keys found for prefix '{module_prefix_with_dot}' in state_dict.")
            return nn.modules.module._IncompatibleKeys([], list(state_dict.keys()))

        # Remove o prefixo do módulo das chaves antes de chamar o super().load_state_dict
        state_dict_local = {
            k.removeprefix(module_prefix_with_dot): v for k, v in relevant_keys.items()
        }

        # Inicializa os pesos apenas se não tiverem sido inicializados e houver chaves
        # e se for um tipo de camada suportado (tem orig_module)
        if not self._initialized and state_dict_local and self._orig_module is not None:
            try:
                self.initialize_weights()
                self._initialized = True  # Marca como inicializado após a tentativa
            except NotImplementedError as e:
                print(
                    f"Erro ao inicializar pesos para {self.prefix}: {e}. O módulo pode não ser carregado corretamente."
                )
                # Não marca como inicializado se a inicialização falhar

        # Tenta carregar o state_dict local apenas se inicializado
        if self._initialized:
            load_result = super().load_state_dict(
                state_dict_local, strict=strict, assign=assign
            )
        else:
            # Se não foi inicializado (ou falhou), retorna indicando que nada foi carregado
            missing_keys = list(state_dict_local.keys())
            load_result = nn.modules.module._IncompatibleKeys(missing_keys, [])

        # Remove as chaves carregadas (ou que deveriam ter sido carregadas)
        # do state_dict original passado para rastrear as restantes
        keys_processed_or_skipped = list(relevant_keys.keys())
        for key in keys_processed_or_skipped:
            if (
                key in state_dict
            ):  # Verifica se a chave ainda existe (pode ter sido pop'ada)
                state_dict.pop(
                    key
                )  # Remove as chaves que foram processadas (ou deveriam ter sido)

        return load_result

    # --- state_dict modificado para incluir prefixo ---
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # Obtém o state_dict local (sem prefixo do módulo)
        local_state_dict = super().state_dict(
            *args, destination=None, prefix="", keep_vars=keep_vars
        )

        # Adiciona o prefixo do módulo (self.prefix) a cada chave local
        state_dict_with_prefix = {
            self.prefix + k: v for k, v in local_state_dict.items()
        }

        # Lida com o argumento 'prefix' padrão do método state_dict do PyTorch
        # (este prefix geralmente vem de chamadas aninhadas, como model.state_dict())
        if prefix:
            state_dict_with_prefix = {
                prefix + k: v for k, v in state_dict_with_prefix.items()
            }

        # Atualiza o destino se fornecido, caso contrário, retorna o dicionário com prefixos
        if destination is None:
            destination = state_dict_with_prefix
        else:
            destination.update(state_dict_with_prefix)

        return destination

    def initialize_weights(self):
        if self._initialized:
            return

        match type(self).__name__:
            case "LoRAModule" | "DoRAModule":
                down, up = self.create_layer()
                self.lora_down = down
                self.lora_up = up
                nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_up.weight)

                if hasattr(self, "dora_scale") and self.dora_scale is None:
                    orig_weight = get_unquantized_weight(self.orig_module, torch.float32, self.train_device)
                    eps = torch.finfo(orig_weight.dtype).eps if self.norm_epsilon else 0.0
                    if isinstance(self.orig_module, nn.Conv2d):
                        norm = torch.norm(orig_weight, p=2, dim=(1, 2, 3), keepdim=True) + eps
                        # self.dora_num_dims não é mais necessário da forma como a normalização foi reescrita
                    elif isinstance(self.orig_module, nn.Linear):
                        norm = torch.norm(orig_weight, p=2, dim=0, keepdim=True) + eps
                        # self.dora_num_dims não é mais necessário
                    else:
                        # Fallback ou erro
                        print(f"Warning: Using generic L2 norm for unsupported layer type {type(self.orig_module)} in DoRA init for {self.prefix}")
                        norm = torch.norm(orig_weight, p=2) + eps
                    
                    self.dora_scale = nn.Parameter(norm.to(self.orig_module.weight.device, dtype=self.orig_module.weight.dtype))
                    del orig_weight

            case "LoHaModule":
                w1b, w1a = self.create_layer()
                w2b, w2a = self.create_layer()

                self.hada_w1_a = w1a.weight
                self.hada_w1_b = w1b.weight
                self.hada_w2_a = w2a.weight
                self.hada_w2_b = w2b.weight

                nn.init.normal_(self.hada_w1_a, std=0.1)
                nn.init.normal_(self.hada_w1_b, std=1)
                nn.init.constant_(self.hada_w2_a, 0)
                nn.init.normal_(self.hada_w2_b, std=1)

            case _:
                raise NotImplementedError(
                    f"initialize_weights() não implementado para {type(self).__name__}"
                )

        self._initialized = True

    @abstractmethod
    def apply_to_module(self):
        pass

    @abstractmethod
    def extract_from_module(self, base_module: nn.Module):
        pass

    def create_layer(self) -> tuple[nn.Module, nn.Module]:
        """Generic helper function for creating a PEFT layer, like LoRA.

        Creates down/up layer modules for the given layer type in the
        orig_module, for the given rank.

        Does not perform initialization, as that usually depends on the PEFT
        method.
        """
        if self.orig_module is None:
            raise RuntimeError(
                f"Cannot create layer for {self.prefix}: Original module is None."
            )

        device = self.orig_module.weight.device
        match self.orig_module:
            case nn.Linear():
                in_features = self.orig_module.in_features
                out_features = self.orig_module.out_features
                # Certifica-se que self.rank existe
                if not hasattr(self, "rank"):
                    raise AttributeError(f"Rank not set for {self.prefix}")
                lora_down = nn.Linear(in_features, self.rank, bias=False, device=device)
                lora_up = nn.Linear(self.rank, out_features, bias=False, device=device)

            case nn.Conv2d():
                in_channels = self.orig_module.in_channels
                out_channels = self.orig_module.out_channels
                kernel_size = self.orig_module.kernel_size
                stride = self.orig_module.stride
                padding = self.orig_module.padding
                dilation = self.orig_module.dilation
                groups = self.orig_module.groups
                # Certifica-se que self.rank existe
                if not hasattr(self, "rank"):
                    raise AttributeError(f"Rank not set for {self.prefix}")
                lora_down = Conv2d(
                    in_channels,
                    self.rank,
                    kernel_size,
                    stride,
                    padding,
                    dilation=dilation,
                    bias=False,
                    device=device,
                )
                lora_up = Conv2d(
                    self.rank,
                    out_channels // groups,
                    (1, 1),
                    stride=1,
                    padding=0,
                    bias=False,
                    device=device,
                )  # Stride e padding explícitos

            case _:
                raise NotImplementedError(
                    f"Layer creation not implemented for type: {type(self.orig_module)}"
                )

        return lora_down, lora_up

    @classmethod
    def make_dummy(cls):
        """Create a dummy version of a PEFT class."""

        class Dummy(cls):
            def __init__(self, *args, **kwargs):
                prefix = args[0] if args else kwargs.get("prefix", "dummy_prefix")
                PeftBase.__init__(
                    self, prefix, None
                )  # Chama __init__ da base sem módulo real
                self._state_dict = {}
                self._initialized = (
                    False  # Dummies não são 'inicializados' no sentido de ter pesos
                )

            def forward(self, *args, **kwargs):
                raise NotImplementedError(
                    "Dummy module should not perform forward pass."
                )

            def load_state_dict(
                self,
                state_dict: Mapping[str, Any],
                strict: bool = True,
                assign: bool = False,
            ):
                module_prefix_with_dot = self.prefix
                relevant_keys = {
                    k: v
                    for k, v in state_dict.items()
                    if k.startswith(module_prefix_with_dot)
                }
                state_dict_local = {
                    k.removeprefix(module_prefix_with_dot): v
                    for k, v in relevant_keys.items()
                }

                if not state_dict_local:
                    return nn.modules.module._IncompatibleKeys([], [])

                self._initialized = (
                    True  # Marca como 'inicializado' para ter um state_dict
                )
                self._state_dict = copy.deepcopy(state_dict_local)

                keys_to_remove = list(relevant_keys.keys())
                for key in keys_to_remove:
                    if key in state_dict:
                        state_dict.pop(key)

                return nn.modules.module._IncompatibleKeys([], [])

            def state_dict(
                self, *args, destination=None, prefix="", keep_vars=False
            ):  # Assinatura completa
                if not self._initialized:
                    # Retorna vazio se nunca carregou nada
                    if destination is None:
                        destination = {}
                    return destination

                # Adiciona o prefixo do módulo (self.prefix)
                state_dict_with_module_prefix = {
                    self.prefix + k: v for k, v in self._state_dict.items()
                }

                # Adiciona o prefixo externo (geralmente da chamada pai)
                if prefix:
                    state_dict_with_full_prefix = {
                        prefix + k: v for k, v in state_dict_with_module_prefix.items()
                    }
                else:
                    state_dict_with_full_prefix = state_dict_with_module_prefix

                if destination is None:
                    destination = state_dict_with_full_prefix
                else:
                    destination.update(state_dict_with_full_prefix)

                return destination

            def initialize_weights(self):
                pass  # Não faz nada

            def hook_to_module(self):
                raise NotImplementedError("Should never be called on a dummy module.")

            def remove_hook_from_module(self):
                raise NotImplementedError("Should never be called on a dummy module.")

            def apply_to_module(self):
                raise NotImplementedError("Should never be called on a dummy module.")

            def extract_from_module(self, base_module: nn.Module):
                raise NotImplementedError("Should never be called on a dummy module.")

        return Dummy


class LoHaModule(PeftBase):
    """Implementation of LoHa from Lycoris."""

    rank: int
    dropout: Dropout
    hada_w1_a: Parameter | None
    hada_w1_b: Parameter | None
    hada_w2_a: Parameter | None
    hada_w2_b: Parameter | None

    def __init__(
        self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float
    ):
        super().__init__(prefix, orig_module)
        self.rank = rank
        self.dropout = Dropout(0)
        self.register_buffer("alpha", torch.tensor(alpha))
        self.hada_w1_a = None
        self.hada_w1_b = None
        self.hada_w2_a = None
        self.hada_w2_b = None

        if orig_module is not None:
            try:  # Tenta inicializar, mas permite falha se o tipo de camada não for suportado
                self.initialize_weights()
                self.alpha = self.alpha.to(orig_module.weight.device)
            except NotImplementedError:
                print(
                    f"Aviso: Não foi possível inicializar LoHa para {prefix} devido ao tipo de camada."
                )
        # self.alpha.requires_grad_(False)

    def check_initialized(self):
        super().check_initialized()
        assert self.hada_w1_a is not None
        assert self.hada_w1_b is not None
        assert self.hada_w2_a is not None
        assert self.hada_w2_b is not None

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        if self.op is None:  # Se a camada original não era suportada
            print(
                f"Aviso: Pulando forward de LoHa para {self.prefix} (tipo de camada não suportado). Retornando saída original."
            )
            return self.orig_forward(x)

        W1 = self.make_weight(
            self.dropout(self.hada_w1_b), self.dropout(self.hada_w1_a)
        )
        W2 = self.make_weight(
            self.dropout(self.hada_w2_b), self.dropout(self.hada_w2_a)
        )
        W = (W1 * W2) * (self.alpha.item() / self.rank)  # Usa .item()
        return self.orig_forward(x) + self.op(x, W, bias=None, **self.layer_kwargs)

    def apply_to_module(self):
        pass  # TODO

    def extract_from_module(self, base_module: nn.Module):
        pass  # TODO


class LoRAModule(PeftBase):
    lora_down: nn.Module | None
    lora_up: nn.Module | None
    rank: int
    alpha: torch.Tensor
    dropout: Dropout

    def __init__(
        self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float
    ):
        super().__init__(prefix, orig_module)
        self.rank = rank
        self.dropout = Dropout(0)
        self.register_buffer("alpha", torch.tensor(alpha))
        self.lora_down = None
        self.lora_up = None

        if orig_module is not None:
            try:  # Tenta inicializar, mas permite falha se o tipo de camada não for suportado
                self.initialize_weights()
                self.alpha = self.alpha.to(orig_module.weight.device)
            except NotImplementedError:
                print(
                    f"Aviso: Não foi possível inicializar LoRA para {prefix} devido ao tipo de camada."
                )
        # self.alpha.requires_grad_(False)

    def check_initialized(self):
        super().check_initialized()
        assert (
            self.lora_down is not None
        ), f"LoRA down not initialized for {self.prefix}"
        assert self.lora_up is not None, f"LoRA up not initialized for {self.prefix}"

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        if self.op is None:  # Se a camada original não era suportada
            print(
                f"Aviso: Pulando forward de LoRA para {self.prefix} (tipo de camada não suportado). Retornando saída original."
            )
            return self.orig_forward(x)

        lora_output = self.lora_up(self.lora_down(self.dropout(x)))
        original_output = self.orig_forward(x)
        scale = self.alpha.item() / self.rank
        return original_output + lora_output * scale

    def apply_to_module(self):
        pass  # TODO

    def extract_from_module(self, base_module: nn.Module):
        pass  # TODO


class DoRAModule(LoRAModule):
    """Weight-decomposed low rank adaptation."""

    dora_num_dims: int
    dora_scale: Parameter | None
    norm_epsilon: bool
    train_device: torch.device

    def __init__(self, *args, **kwargs):
        self.dora_scale = None
        self.norm_epsilon = kwargs.pop("norm_epsilon", False)
        self.train_device = kwargs.pop("train_device", torch.device("cpu"))
        super().__init__(*args, **kwargs)  # Chama __init__ de LoRAModule

    def check_initialized(self):
        super().check_initialized()
        assert (
            self.dora_scale is not None
        ), f"DoRA scale not initialized for {self.prefix}"

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        if self.op is None:  # Se a camada original não era suportada
            print(
                f"Aviso: Pulando forward de DoRA para {self.prefix} (tipo de camada não suportado). Retornando saída original."
            )
            return self.orig_forward(x)

        A = self.lora_down.weight
        B = self.lora_up.weight
        orig_weight = get_unquantized_weight(self.orig_module, A.dtype, self.train_device)
        lora_delta_w = self.make_weight(A, B) * (self.alpha.item() / self.rank)
        WP = orig_weight + lora_delta_w
        del orig_weight, lora_delta_w # Libera memória

        eps = torch.finfo(WP.dtype).eps if self.norm_epsilon else 0.0
        
        if isinstance(self.orig_module, nn.Conv2d):
            # Norma por filtro de saída (canal de saída)
            norm = torch.norm(WP.detach().to(torch.float32), p=2, dim=(1, 2, 3), keepdim=True) + eps
        elif isinstance(self.orig_module, nn.Linear):
            # Norma por neurônio de saída (coluna)
            norm = torch.norm(WP.detach().to(torch.float32), p=2, dim=0, keepdim=True) + eps
        else:
            # Fallback ou erro se outro tipo de camada for encontrado inesperadamente
            # Usando norma L2 geral como fallback, mas pode não ser ideal.
            print(f"Warning: Using generic L2 norm for unsupported layer type {type(self.orig_module)} in DoRA forward for {self.prefix}")
            norm = torch.norm(WP.detach().to(torch.float32), p=2) + eps

        WP_normalized = WP / norm.to(WP.device, dtype=WP.dtype) # Garante que a norma esteja no dispositivo/dtype correto
        WP_scaled = self.dora_scale * WP_normalized

        x_dropout = self.dropout(x)
        return self.op(x_dropout, WP_scaled, self.orig_module.bias, **self.layer_kwargs)


# Cria instâncias Dummy após as definições das classes originais
DummyLoRAModule = LoRAModule.make_dummy()
DummyDoRAModule = DoRAModule.make_dummy()
DummyLoHaModule = LoHaModule.make_dummy()


class LoRAModuleWrapper:
    """
    Gerencia módulos PEFT (LoRA, LoHa, DoRA) para um módulo PyTorch original.

    Permite aplicar diferentes configurações de rank e alpha para diferentes
    camadas com base em padrões de nome definidos em `config.lora_layer_patterns`.
    Também suporta um `module_filter` para pré-selecionar camadas.
    """

    orig_module: nn.Module
    prefix: str
    peft_type: Any  # Tipo PeftType ou placeholder
    default_rank: int  # Rank padrão global
    default_alpha: float  # Alpha padrão global
    lora_layer_patterns: dict[str, dict[str, int | float]]  # Configs por padrão
    ruleset: RuleSet
    _sorted_patterns: list[
        str
    ]  # Chaves de lora_layer_patterns ordenadas por especificidade
    klass: type[PeftBase]  # Classe PEFT a ser usada (LoRA, LoHa, DoRA)
    dummy_klass: type[PeftBase]  # Classe Dummy correspondente
    global_additional_kwargs: dict  # Kwargs para DoRA (norm_epsilon, train_device)
    lora_modules: dict[str, PeftBase]  # Módulos PEFT gerenciados (reais e dummies)

    def __init__(
        self,
        orig_module: nn.Module | None,
        prefix: str,
        config: TrainConfig,
        # INÍCIO ALTERAÇÃO - Modificado para aceitar nome do preset
        preset_name: str | None = None,  # Nome do preset a ser carregado de PRESETS
        external_module_filter: (
            list[str] | None
        ) = None,  # Filtro adicional/override externo
        # FIM ALTERAÇÃO
        module_filter: list[str] | None = None,
    ):
        """
        Inicializa o Wrapper.

        Args:
            orig_module: O módulo nn.Module original a ser adaptado.
            prefix: Prefixo a ser adicionado às chaves LoRA (ex: "lora_unet").
            config: Objeto TrainConfig contendo as configurações:
                - peft_type: Tipo de adaptação (LoRA, LoHa, DoRA).
                - lora_rank: Rank padrão a ser usado se nenhum padrão corresponder.
                - lora_alpha: Alpha padrão a ser usado se nenhum padrão corresponder.
                - lora_layer_patterns: Dicionário mapeando padrões de nome de camada
                  para dicionários com 'rank' e/ou 'alpha' específicos.
                  Ex: {"attn1": {"rank": 16}, "ff.net": {"rank": 32, "alpha": 16}}
                - lora_decompose: True para usar DoRA em vez de LoRA.
                - lora_decompose_norm_epsilon: Epsilon para DoRA.
                - train_device: Dispositivo para DoRA.
            module_filter: Lista opcional de substrings. Apenas camadas cujo nome
              contém pelo menos uma dessas substrings serão consideradas para PEFT.
              Se None ou vazio, todas as camadas Linear/Conv2d são consideradas.
        """
        if orig_module is None:
            print("Aviso: LoRAModuleWrapper inicializado sem um módulo original.")
            # Pode ser útil para carregar apenas a partir de um state_dict,
            # mas a criação inicial de módulos não ocorrerá.
        self.orig_module = orig_module
        self.prefix = prefix
        self.peft_type = config.peft_type
        self.config = config

        # Armazena os valores padrão
        self.default_rank = config.lora_rank
        self.default_alpha = config.lora_alpha

        # INÍCIO ALTERAÇÃO - Lógica Refatorada para Presets, Filtros e Blacklist
        # 1. Carregar padrões e filtros do preset
        from modules.modelSetup.StableDiffusionXLLoRASetup import (
            PRESETS,
        )  # Mover import para cá é mais seguro se PRESETS for complexo

        # Começa com os padrões globais da config principal
        self.lora_layer_patterns = (
            config.lora_layer_patterns.copy() if config.lora_layer_patterns else {}
        )
        preset_inclusion_patterns = []  # Padrões de *inclusão* vindos do preset

        if preset_name:
            preset_config = PRESETS.get(preset_name)
            if (
                preset_config is None and preset_name != "full"
            ):  # Trata preset não encontrado ou explicitamente None
                print(
                    f"[LoRA WARNING] Preset '{preset_name}' não encontrado ou é None. Usando padrões globais e filtro externo/blacklist."
                )
            elif preset_name == "full":
                print(
                    f"[LoRA INFO] Preset 'full' selecionado. Nenhum filtro de inclusão ou override de rank/alpha do preset aplicado."
                )
            elif isinstance(preset_config, dict):
                print(
                    f"[LoRA INFO] Usando preset '{preset_name}' com {len(preset_config)} padrões/overrides."
                )
                # Padrões de inclusão são as *chaves* do dicionário do preset
                preset_inclusion_patterns = list(preset_config.keys())
                # Mescla os overrides de rank/alpha do preset sobre os globais
                self.lora_layer_patterns.update(preset_config)
            else:
                # Caso inesperado (e.g., lista em vez de dict) - tratar como erro ou aviso
                print(
                    f"[LoRA WARNING] Preset '{preset_name}' tem formato inesperado ({type(preset_config)}). Ignorando preset."
                )
        else:
            print(
                "[LoRA INFO] Nenhum preset especificado. Usando padrões globais e filtro externo/blacklist."
            )

        # 2. Combinar filtros de *inclusão* (Preset + External)
        # Se um módulo corresponder a *qualquer* padrão de inclusão, ele é considerado.
        combined_inclusion_patterns = set(preset_inclusion_patterns)
        if external_module_filter:
            combined_inclusion_patterns.update(external_module_filter)

        # Lista final de padrões de inclusão (vazia significa "incluir tudo")
        self.inclusion_filter_patterns = [
            p.strip() for p in combined_inclusion_patterns if p.strip()
        ]
        print(
            f"[LoRA INFO] Padrões de inclusão ativos: {self.inclusion_filter_patterns if self.inclusion_filter_patterns else 'Nenhum (incluir tudo)'}"
        )

        # 3. Preparar filtro de *exclusão* (Blacklist)
        self.exclusion_filter_patterns = [
            b.strip() for b in (config.lora_layers_blacklist or []) if b.strip()
        ]
        print(
            f"[LoRA INFO] Padrões de exclusão (blacklist) ativos: {self.exclusion_filter_patterns if self.exclusion_filter_patterns else 'Nenhum'}"
        )

        # 4. Inicializar RuleSet com os padrões *finais* (global + preset overrides)
        # A RuleSet será usada APENAS para determinar rank/alpha dos módulos que passarem nos filtros.
        self.ruleset = RuleSet(self.lora_layer_patterns)
        print(
            f"[LoRA INFO] RuleSet inicializada com {len(self.ruleset.patterns)} regras de rank/alpha."
        )
        # Debug: Mostrar algumas regras carregadas
        # print(f"[LoRA DEBUG] Primeiras 5 regras na RuleSet: {list(self.ruleset.patterns.items())[:5]}")

        # 5. Definir classes PEFT e kwargs globais (sem alteração aqui)
        self.global_additional_kwargs = {}
        if self.peft_type == PeftType.LORA:
            if config.lora_decompose:  # Usar config diretamente
                self.klass = DoRAModule
                self.dummy_klass = DummyDoRAModule
                self.global_additional_kwargs = {
                    "norm_epsilon": config.lora_decompose_norm_epsilon,
                    "train_device": torch.device(
                        config.train_device
                    ),  # Usar config diretamente
                }
            else:
                self.klass = LoRAModule
                self.dummy_klass = DummyLoRAModule
        elif self.peft_type == PeftType.LOHA:
            self.klass = LoHaModule
            self.dummy_klass = DummyLoHaModule
        else:
            raise ValueError(f"Unsupported PeftType: {self.peft_type}")

        # 6. Criar os módulos PEFT (agora com lógica de filtragem robusta)
        # A criação foi movida para um método separado para clareza
        self.lora_modules = self._initialize_peft_modules(orig_module)
        # FIM ALTERAÇÃO

    def _should_include_module(self, module_name: str) -> bool:
        """Verifica se um nome de módulo deve ser incluído com base nos filtros."""
        # 1. Verificar filtro de inclusão
        passes_inclusion = False
        if not self.inclusion_filter_patterns:
            passes_inclusion = True  # Se não há filtros de inclusão, passa tudo
        else:
            for pattern in self.inclusion_filter_patterns:
                if fnmatch.fnmatch(module_name, pattern):
                    passes_inclusion = True
                    break
        if not passes_inclusion:
            return False  # Falhou na inclusão

        # 2. Verificar filtro de exclusão (blacklist)
        for pattern in self.exclusion_filter_patterns:
            if fnmatch.fnmatch(module_name, pattern):
                # print(f"[LoRA FILTER DEBUG] Module '{module_name}' EXCLUDED by blacklist pattern '{pattern}'") # Debug Opcional
                return False  # Foi explicitamente excluído

        # Se passou pela inclusão (se houver) e não foi excluído, inclui.
        # print(f"[LoRA FILTER DEBUG] Module '{module_name}' INCLUDED") # Debug Opcional
        return True

    # INÍCIO ALTERAÇÃO - Método de inicialização refatorado
    def _initialize_peft_modules(
        self, root_module: nn.Module | None
    ) -> dict[str, PeftBase]:
        """
        Identifica, filtra e cria os módulos PEFT reais.
        """
        lora_modules = {}
        if root_module is None:
            print(
                "[LoRA WARNING] _initialize_peft_modules chamado sem root_module. Nenhum módulo será criado."
            )
            return lora_modules

        print("[LoRA INFO] Iniciando identificação e criação de módulos PEFT...")
        modules_to_create = []

        # 1. Identificar todos os candidatos e aplicar filtros
        for name, child_module in root_module.named_modules():
            if not isinstance(child_module, (Linear, Conv2d)):
                continue  # Pula tipos não suportados

            # Aplica a lógica de filtragem combinada (inclusão + exclusão)
            if self._should_include_module(name):
                modules_to_create.append((name, child_module))
            # else: # Debug opcional
            #     print(f"[LoRA FILTER] Skipping module: {name}")

        print(
            f"[LoRA INFO] Identificados {len(modules_to_create)} módulos candidatos após filtragem."
        )

        # 2. Criar módulos PEFT para os candidatos filtrados
        for name, child_module in modules_to_create:
            # Constrói o prefixo completo para o módulo PEFT
            # (Lógica ligeiramente simplificada para evitar duplicação de prefixo se já presente)
            clean_name = name
            full_peft_prefix = f"{self.prefix}_{clean_name}"

            # Determina Rank e Alpha usando RuleSet (baseado no nome original 'name')
            rank_to_use = self.default_rank
            alpha_to_use = self.default_alpha
            matched_config = self.ruleset.match(name)  # Usa o nome original da camada

            if matched_config:
                rank_to_use = matched_config.get("rank", self.default_rank)
                alpha_to_use = matched_config.get("alpha", self.default_alpha)
                # print(f"[LoRA RULE DEBUG] Match for '{name}': {matched_config} -> Rank={rank_to_use}, Alpha={alpha_to_use}") # Debug opcional

            # Prepara args e kwargs para a classe PEFT
            args_for_this_module = [rank_to_use, alpha_to_use]
            kwargs_for_this_module = self.global_additional_kwargs.copy()

            # Cria a instância
            try:
                # Debug log mais informativo
                match_info = "(Matched Rule)" if matched_config else "(Default)"
                log_msg = (
                    f"[LoRA CREATE] Creating {self.klass.__name__} for: {full_peft_prefix} "
                    f"(Layer: {name}) {match_info} -> Rank={rank_to_use}, Alpha={alpha_to_use}"
                )
                # print(log_msg) # Debug no console
                # with open("lora_creation_log.txt", "a", encoding="utf-8") as debug_file: # Log em arquivo opcional
                #     debug_file.write(log_msg + "\n")

                lora_modules[name] = self.klass(
                    full_peft_prefix,
                    child_module,
                    *args_for_this_module,
                    **kwargs_for_this_module,
                )

            except Exception as e:
                print(
                    f"[LoRA ERROR] Erro ao criar módulo PEFT para {name} (prefixo {full_peft_prefix}) "
                    f"com rank={rank_to_use}, alpha={alpha_to_use}: {e}"
                )

        print(
            f"[LoRA INFO] Criados {len(lora_modules)} módulos PEFT para o prefixo '{self.prefix}'."
        )
        return lora_modules

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        """
        Carrega o state_dict nos módulos PEFT gerenciados.

        Primeiro tenta carregar nos módulos PEFT reais existentes.
        As chaves restantes no state_dict que correspondem ao prefixo
        são usadas para criar módulos Dummy, usando os ranks/alphas padrão
        para a instanciação do Dummy.
        """
        remaining_state_dict = copy.deepcopy(state_dict)
        loaded_prefixes = set()  # Rastreia prefixos já processados (reais ou dummy)

        # 1. Tenta carregar nos módulos reais existentes
        # Usa items() para iterar sobre cópia, permitindo modificar o dict original se necessário
        current_real_modules = {
            k: v
            for k, v in self.lora_modules.items()
            if not isinstance(v, self.dummy_klass)
        }
        for name, module in current_real_modules.items():
            try:
                # load_state_dict da PeftBase já lida com o prefixo interno
                module.load_state_dict(
                    remaining_state_dict
                )  # Modifica remaining_state_dict in-place
                loaded_prefixes.add(module.prefix)
            except Exception as e:
                print(
                    f"Erro ao carregar state_dict para módulo real {name} (prefixo {module.prefix}): {e}"
                )

        # 2. Processa chaves remanescentes para criar dummies
        # Filtra apenas chaves que começam com o prefixo geral do wrapper
        potential_dummy_keys = {
            k: v for k, v in remaining_state_dict.items() if k.startswith(self.prefix)
        }

        while potential_dummy_keys:
            # Pega uma chave exemplo para determinar o prefixo do módulo dummy
            key_example = next(iter(potential_dummy_keys))

            # Lógica para extrair o prefixo do módulo PEFT da chave do parâmetro
            parts = key_example.split(".")
            dummy_prefix_parts = []
            known_suffix = False
            # Sufixos comuns de parâmetros PEFT
            common_suffixes = [
                ".lora_down.weight",
                ".lora_up.weight",
                ".alpha",
                ".dora_scale",
                ".hada_w1_a",
                ".hada_w1_b",
                ".hada_w2_a",
                ".hada_w2_b",
            ]
            for suffix in common_suffixes:
                if key_example.endswith(suffix):
                    # Remove o sufixo (ex: '.lora_down.weight')
                    num_parts_to_remove = suffix.count(".") + 1
                    dummy_prefix_parts = parts[:-num_parts_to_remove]
                    known_suffix = True
                    break

            if not known_suffix:
                # Não conseguiu identificar um sufixo PEFT conhecido
                # print(f"Aviso: Não foi possível determinar o prefixo do módulo para a chave remanescente: {key_example}. Pulando esta chave.")
                potential_dummy_keys.pop(key_example)  # Remove a chave problemática
                continue  # Tenta a próxima chave remanescente

            # Monta o prefixo completo do módulo dummy (termina com '.')
            dummy_module_prefix_with_dot = ".".join(dummy_prefix_parts) + "."

            # Se já carregamos um módulo real ou criamos um dummy com este prefixo,
            # apenas removemos todas as chaves associadas a ele e continuamos
            if dummy_module_prefix_with_dot in loaded_prefixes:
                keys_to_remove = [
                    k
                    for k in potential_dummy_keys
                    if k.startswith(dummy_module_prefix_with_dot)
                ]
                for k in keys_to_remove:
                    if k in potential_dummy_keys:
                        potential_dummy_keys.pop(k)
                continue  # Passa para a próxima chave remanescente

            # Determina o 'nome' relativo do módulo dummy (usado como chave em self.lora_modules)
            # Ex: "lora_unet_down_blocks_0_attn_0." -> "down_blocks_0_attn_0"
            dummy_module_name = (
                dummy_module_prefix_with_dot.removeprefix(
                    self.prefix + "_"
                ).removesuffix(".")
                if dummy_module_prefix_with_dot.startswith(self.prefix + "_")
                else ""
            )

            # Cria a instância Dummy
            # Dummies não precisam de rank/alpha específicos, usamos os padrões globais.
            dummy_args = [self.default_rank, self.default_alpha]
            dummy_kwargs = self.global_additional_kwargs.copy()
            try:
                # print(f"Creating dummy for prefix: {dummy_module_prefix_with_dot} (name: {dummy_module_name})") # Debug
                dummy_module = self.dummy_klass(
                    dummy_module_prefix_with_dot,  # Passa o prefixo correto
                    None,  # Módulo original é None para dummies
                    *dummy_args,
                    **dummy_kwargs,
                )
                # Passa uma cópia do dict de chaves potenciais para o dummy processar
                dummy_module.load_state_dict(
                    copy.deepcopy(potential_dummy_keys)
                )  # Modifica o dict passado in-place

                # Remove as chaves que o dummy efetivamente carregou do nosso dict principal
                # (O state_dict do dummy retorna as chaves com o prefixo correto)
                dummy_keys_loaded = list(dummy_module.state_dict().keys())
                for k in dummy_keys_loaded:
                    if k in potential_dummy_keys:
                        potential_dummy_keys.pop(
                            k
                        )  # Remove do dict que estamos iterando

                # Adiciona o dummy ao gerenciador
                self.lora_modules[dummy_module_name] = dummy_module
                # Marca este prefixo como processado para evitar recriação
                loaded_prefixes.add(dummy_module_prefix_with_dot)

            except Exception as e:
                print(
                    f"Erro ao criar ou carregar dummy para prefixo {dummy_module_prefix_with_dot}: {e}"
                )
                # Remove todas as chaves associadas a este prefixo para evitar tentar de novo
                keys_to_remove = [
                    k
                    for k in potential_dummy_keys
                    if k.startswith(dummy_module_prefix_with_dot)
                ]
                for k in keys_to_remove:
                    if k in potential_dummy_keys:
                        potential_dummy_keys.pop(k)

        # Verifica se sobrou alguma chave não utilizada no final
        final_remaining_keys = list(potential_dummy_keys.keys())
        if final_remaining_keys:
            # Pode ser útil logar isso para debug
            print(
                f"Aviso: Chaves não utilizadas encontradas após carregar state dict para prefixo '{self.prefix}': {final_remaining_keys}"
            )
            pass

    def state_dict(self) -> dict:
        """Retorna o state dict contendo chaves de todos os módulos gerenciados (reais e dummy)."""
        state_dict = {}
        for module in self.lora_modules.values():
            # state_dict de PeftBase/Dummy já adiciona o prefixo do módulo
            module_sd = module.state_dict()
            state_dict.update(module_sd)
        return state_dict

    def parameters(self) -> list[Parameter]:
        """Retorna uma lista de parâmetros treináveis de todos os módulos PEFT reais."""
        parameters = []
        for module in self.lora_modules.values():
            # Inclui apenas parâmetros de módulos reais (não dummies)
            if not isinstance(module, self.dummy_klass):
                # Garante que o módulo foi inicializado antes de pegar parâmetros
                if module._initialized:
                    parameters.extend(list(module.parameters()))
        return parameters

    def requires_grad_(self, requires_grad: bool):
        """Define requires_grad para todos os parâmetros de módulos PEFT reais."""
        for module in self.lora_modules.values():
            if not isinstance(module, self.dummy_klass) and module._initialized:
                module.requires_grad_(requires_grad)

    def to(
        self, device: torch.device = None, dtype: torch.dtype = None
    ) -> "LoRAModuleWrapper":
        """Move todos os módulos gerenciados (reais e dummy) para o dispositivo/dtype."""
        for module in self.lora_modules.values():
            # Dummies podem não ter parâmetros, mas a chamada a `to` é segura
            module.to(device, dtype)
        return self

    def modules(self) -> list[nn.Module]:
        """Retorna uma lista de todos os módulos gerenciados (reais e dummy)."""
        return list(self.lora_modules.values())

    def hook_to_module(self):
        """Aplica o hook de forward nos módulos originais correspondentes aos módulos PEFT reais."""
        if self.orig_module is None:
            print("Aviso: Não é possível aplicar hooks sem o módulo original.")
            return
        hook_count = 0
        for name, module in self.lora_modules.items():
            if not isinstance(module, self.dummy_klass):
                try:
                    # A PeftBase armazena a referência ao módulo original
                    module.hook_to_module()
                    hook_count += 1
                except Exception as e:
                    print(f"Erro ao aplicar hook para {name}: {e}")
        # print(f"Aplicados hooks para {hook_count} módulos PEFT.")

    def remove_hook_from_module(self):
        """Remove o hook de forward dos módulos originais."""
        if self.orig_module is None:
            # print("Aviso: Não é possível remover hooks sem o módulo original (ou já foram removidos).")
            return
        remove_count = 0
        for name, module in self.lora_modules.items():
            if not isinstance(module, self.dummy_klass):
                try:
                    module.remove_hook_from_module()
                    remove_count += 1
                except Exception as e:
                    print(f"Erro ao remover hook para {name}: {e}")
        # print(f"Removidos hooks de {remove_count} módulos PEFT.")

    def apply_to_module(self):
        """Aplica (merge) os pesos PEFT diretamente nos pesos dos módulos originais (apenas módulos reais)."""
        if self.orig_module is None:
            print("Aviso: Não é possível aplicar pesos sem o módulo original.")
            return
        apply_count = 0
        for name, module in self.lora_modules.items():
            if not isinstance(module, self.dummy_klass):
                try:
                    module.apply_to_module()  # A implementação está em PeftBase/subclasses
                    apply_count += 1
                except NotImplementedError:
                    print(
                        f"Aviso: apply_to_module não implementado para {type(module).__name__} ({name})."
                    )
                except Exception as e:
                    print(f"Erro ao aplicar pesos para {name}: {e}")
        # print(f"Pesos aplicados para {apply_count} módulos PEFT.")

    def extract_from_module(self, base_module: nn.Module):
        """
        Extrai pesos PEFT comparando o módulo original atual com um módulo base.
        (Requer implementação em LoRAModule, DoRAModule, etc.)
        """
        if self.orig_module is None:
            print("Aviso: Não é possível extrair pesos sem o módulo original atual.")
            return
        extract_count = 0
        for name, module in self.lora_modules.items():
            if not isinstance(module, self.dummy_klass):
                try:
                    # Encontra o submódulo correspondente no modelo base fornecido
                    corresponding_base_submodule = base_module.get_submodule(name)
                    module.extract_from_module(
                        corresponding_base_submodule
                    )  # Implementação em PeftBase/subclasses
                    extract_count += 1
                except AttributeError:
                    print(
                        f"Aviso: Não foi possível encontrar o submódulo base '{name}' durante a extração."
                    )
                except NotImplementedError:
                    print(
                        f"Aviso: extract_from_module não implementado para {type(module).__name__} ({name})."
                    )
                except Exception as e:
                    print(f"Erro ao extrair pesos para {name}: {e}")
        # print(f"Pesos extraídos para {extract_count} módulos PEFT.")

    def prune(self):
        """Remove todos os módulos dummy do gerenciamento."""
        initial_count = len(self.lora_modules)
        self.lora_modules = {
            k: v
            for (k, v) in self.lora_modules.items()
            if not isinstance(v, self.dummy_klass)
        }
        pruned_count = initial_count - len(self.lora_modules)
        # print(f"Removidos {pruned_count} módulos dummy. Módulos restantes: {len(self.lora_modules)}")

    def set_dropout(self, dropout_probability: float):
        """Define a probabilidade de dropout para todos os módulos PEFT reais."""
        if not 0 <= dropout_probability <= 1:
            raise ValueError("Probabilidade de dropout deve estar entre 0 e 1")
        for module in self.lora_modules.values():
            if not isinstance(module, self.dummy_klass):
                # Verifica se o módulo tem o atributo dropout e se é um nn.Dropout
                if hasattr(module, "dropout") and isinstance(
                    getattr(module, "dropout", None), nn.Dropout
                ):
                    module.dropout.p = dropout_probability

    # --- Geração de Arquivo de Chaves por Bloco (Mantida para Organização) ---
    @staticmethod
    def get_block_id_from_key_prefix(lora_module_prefix_with_dot: str) -> str | None:
        """
        Determina o Block ID (IN00-OUT11, M00, BASE) a partir do prefixo completo.
        Usado apenas para organizar o arquivo de chaves. Retorna None se não mapeado.
        """
        lora_module_prefix = lora_module_prefix_with_dot.removesuffix(".")

        # Tratamento especial para mid_block
        if lora_module_prefix.startswith("lora_unet_mid_block"):
            return "M00"

        # Ordena por comprimento decrescente para especificidade
        sorted_map_prefixes = sorted(KEY_TO_BLOCK_MAPPING.keys(), key=len, reverse=True)

        for map_prefix in sorted_map_prefixes:
            # Compara prefixos completos
            if lora_module_prefix == map_prefix or lora_module_prefix.startswith(
                map_prefix + "."
            ):
                # O mapeamento já inclui 'lora_unet_'
                return KEY_TO_BLOCK_MAPPING[map_prefix]

        # Tenta mapear apenas a parte final se não começou com lora_unet_
        # (Menos comum, mas pode ocorrer dependendo do prefixo inicial)
        canonical_prefix = lora_module_prefix.replace(".", "_")
        for map_prefix_short, block_id in KEY_TO_BLOCK_MAPPING.items():
            if canonical_prefix.startswith(map_prefix_short):
                return block_id

        # Se não encontrou correspondência exata ou de prefixo
        # Verifica se pertence a TEXT ENCODER (se aplicável)
        if lora_module_prefix.startswith("lora_te"):  # Ajustar conforme prefixo do TE
            # Poderia retornar um ID específico como "TE" ou agrupar em BASE
            return "BASE"  # Ou "TE" se BLOCKID26 incluir

        # print(f"Aviso: Prefixo do módulo não mapeado para bloco UNet/TE: {lora_module_prefix}. Assumindo BASE.")
        return "BASE"  # Fallback para chaves não mapeadas

    def generate_keys_by_block_file(
        self, output_filename="unetKeysByBlock_generated.txt"
    ):
        """Gera um arquivo de texto com todas as chaves PEFT organizadas por bloco UNet."""
        if not self.lora_modules:
            # print("Aviso: Nenhum módulo LoRA/DoRA/LoHa encontrado. Arquivo de chaves não gerado.")
            return

        # print(f"Gerando arquivo de chaves por bloco: {output_filename}")
        # Usa um dict com default list para simplificar
        from collections import defaultdict

        blocks_data: dict[str, list[str]] = defaultdict(list)
        all_block_ids = set(BLOCKID26)  # Conjunto para referência

        # Itera sobre os módulos gerenciados (reais e dummies)
        for module_instance in self.lora_modules.values():
            # Pega o prefixo correto armazenado na instância do módulo PEFT
            module_prefix_with_dot = module_instance.prefix
            # Mapeia o prefixo para um ID de bloco (IN00, M00, BASE, etc.)
            block_id = self.get_block_id_from_key_prefix(module_prefix_with_dot)

            if block_id is None:  # Se get_block_id retornar None
                block_id = "BASE"  # Atribui a BASE como fallback

            all_block_ids.add(
                block_id
            )  # Garante que todos os IDs encontrados estejam na lista final

            # Pega as chaves deste módulo específico usando seu state_dict()
            try:
                module_state_dict = module_instance.state_dict()
                # Adiciona as chaves (que já incluem o prefixo) à lista do bloco
                for key in module_state_dict.keys():
                    blocks_data[block_id].append(key)
            except Exception as e:
                print(
                    f"Erro ao obter state_dict para o módulo com prefixo {module_prefix_with_dot}: {e}"
                )

        # Ordena a lista final de IDs de bloco (garante ordem consistente)
        # Coloca BASE no início, depois os outros em ordem alfabética/numérica
        sorted_block_ids = sorted(
            list(all_block_ids), key=lambda x: ("0" if x == "BASE" else "1") + x
        )

        # Formata e escreve o arquivo
        output_lines = []
        for block_id in sorted_block_ids:
            output_lines.append(f"=== {block_id} ===")
            # Ordena as chaves dentro de cada bloco
            keys_in_block = sorted(blocks_data[block_id])
            if keys_in_block:
                output_lines.extend(keys_in_block)
            else:
                # Mensagem padrão para blocos vazios
                output_lines.append("(Vazio)")

            output_lines.append("")  # Linha em branco entre blocos

        try:
            # Garante que o diretório de saída exista
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines).strip())
            # print(f"Arquivo '{output_filename}' gerado com sucesso.")
        except Exception as e:
            print(f"Erro ao escrever o arquivo '{output_filename}': {e}")


def make_layer_patterns(
    layers: list[str], rank: int | None = None, alpha: float | None = None
) -> dict[str, dict[str, int | float]]:
    config = {}
    for layer in layers:
        entry = {}
        if rank is not None:
            entry["rank"] = rank
        if alpha is not None:
            entry["alpha"] = alpha
        config[layer] = entry
    return config
