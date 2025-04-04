# --- START OF FILE loraModuleOneTrainer.txt (com geração automática no __init__) ---

import copy
import math
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any
import re # Adicionado
import os # Adicionado

# Tenta importar TrainConfig de forma segura
try:
    from modules.util.config.TrainConfig import TrainConfig
except ImportError:
    print("Aviso: Não foi possível importar TrainConfig. Usando um placeholder.")
    # Cria um placeholder se a importação falhar (útil para análise estática ou ambientes limitados)
    class TrainConfig:
        def __init__(self):
            self.peft_type = PeftType.LORA # Exemplo
            self.lora_rank = 4
            self.lora_alpha = 1.0
            self.lora_decompose = False
            self.lora_decompose_norm_epsilon = False
            self.train_device = "cpu"
            # Adicione outros atributos necessários com valores padrão

# Tenta importar PeftType de forma segura
try:
    from modules.util.enum.ModelType import PeftType
except ImportError:
    print("Aviso: Não foi possível importar PeftType. Usando um placeholder.")
    # Cria um placeholder enum
    from enum import Enum
    class PeftType(Enum):
        LORA = 0
        LOHA = 1
        # Adicione outros tipos se necessário

# Tenta importar funções de utilidade de forma segura
try:
    from modules.util.quantization_util import get_unquantized_weight, get_weight_shape
except ImportError:
    print("Aviso: Não foi possível importar funções de quantization_util. Usando placeholders.")
    # Cria placeholders se a importação falhar
    def get_unquantized_weight(module, dtype, device): return module.weight # Simplificação
    def get_weight_shape(module): return module.weight.shape # Simplificação


import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Conv2d, Dropout, Linear, Parameter

# --- Adicionado: Definição de blocos e mapeamento ---
BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
KEY_TO_BLOCK_MAPPING = {
    "lora_unet_conv_in": "IN00",
    "lora_unet_time_embedding": "IN09", # Agrupando embeddings aqui por convenção
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
    "lora_unet_mid_block": "M00", # Cobre resnets e attentions do mid block
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
        self.prefix = prefix.replace('.', '_') + '.'
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
                    print(f"Aviso: Tipo de módulo não suportado encontrado em PeftBase: {type(orig_module)}. Continuando, mas pode falhar.")
                    self.op = None
                    self.shape = None
                    # raise NotImplementedError("Only Linear and Conv2d are supported layers.")


    def hook_to_module(self):
        if self.orig_module is None: return # Não pode fazer hook sem módulo original
        if not self.is_applied:
            self.orig_forward = self.orig_module.forward
            self.orig_train = self.orig_module.train
            self.orig_eval = self.orig_module.eval
            self.orig_module.forward = self.forward
            self.orig_module.train = self._wrap_train
            self.orig_module.eval = self._wrap_eval
            self.is_applied = True

    def remove_hook_from_module(self):
        if self.orig_module is None or self.orig_forward is None: return
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.orig_module.train = self.orig_train
            self.orig_module.eval = self.orig_eval
            self.is_applied = False

    def _wrap_train(self, mode=True):
        if self.orig_module is None or self.orig_train is None: return
        self.orig_train(mode)
        self.train(mode)

    def _wrap_eval(self):
        if self.orig_module is None or self.orig_eval is None: return
        self.orig_eval()
        self.eval()

    def make_weight(self, A: Tensor, B: Tensor):
        """Layer-type-independent way of creating a weight matrix from LoRA A/B."""
        if self.shape is None:
             raise RuntimeError(f"Cannot make weight for {self.prefix}: shape not determined (likely unsupported layer type).")
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
             raise AttributeError(f"Original module not set for PEFT layer with prefix {self.prefix}")
        return self._orig_module[0]

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        # Filtra o state_dict para conter apenas chaves relevantes para este módulo específico
        module_prefix_with_dot = self.prefix # self.prefix já termina com '.'
        relevant_keys = {k: v for k, v in state_dict.items() if k.startswith(module_prefix_with_dot)}
        if not relevant_keys:
             # Se nenhuma chave relevante for encontrada, talvez o módulo não esteja no state_dict
             # ou o prefixo está incorreto. Retorna chaves vazias para indicar isso.
             # Não inicializa o módulo se não houver chaves para carregar.
             # print(f"Warning: No keys found for prefix '{module_prefix_with_dot}' in state_dict.")
             return nn.modules.module._IncompatibleKeys([], list(state_dict.keys()))

        # Remove o prefixo do módulo das chaves antes de chamar o super().load_state_dict
        state_dict_local = {k.removeprefix(module_prefix_with_dot): v for k, v in relevant_keys.items()}

        # Inicializa os pesos apenas se não tiverem sido inicializados e houver chaves
        # e se for um tipo de camada suportado (tem orig_module)
        if not self._initialized and state_dict_local and self._orig_module is not None:
             try:
                 self.initialize_weights()
                 self._initialized = True # Marca como inicializado após a tentativa
             except NotImplementedError as e:
                 print(f"Erro ao inicializar pesos para {self.prefix}: {e}. O módulo pode não ser carregado corretamente.")
                 # Não marca como inicializado se a inicialização falhar

        # Tenta carregar o state_dict local apenas se inicializado
        if self._initialized:
            load_result = super().load_state_dict(state_dict_local, strict=strict, assign=assign)
        else:
            # Se não foi inicializado (ou falhou), retorna indicando que nada foi carregado
            missing_keys = list(state_dict_local.keys())
            load_result = nn.modules.module._IncompatibleKeys(missing_keys, [])


        # Remove as chaves carregadas (ou que deveriam ter sido carregadas)
        # do state_dict original passado para rastrear as restantes
        keys_processed_or_skipped = list(relevant_keys.keys())
        for key in keys_processed_or_skipped:
            if key in state_dict: # Verifica se a chave ainda existe (pode ter sido pop'ada)
                 state_dict.pop(key) # Remove as chaves que foram processadas (ou deveriam ter sido)

        return load_result

    # --- state_dict modificado para incluir prefixo ---
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # Obtém o state_dict local (sem prefixo do módulo)
        local_state_dict = super().state_dict(*args, destination=None, prefix='', keep_vars=keep_vars)

        # Adiciona o prefixo do módulo (self.prefix) a cada chave local
        state_dict_with_prefix = {self.prefix + k: v for k, v in local_state_dict.items()}

        # Lida com o argumento 'prefix' padrão do método state_dict do PyTorch
        # (este prefix geralmente vem de chamadas aninhadas, como model.state_dict())
        if prefix:
            state_dict_with_prefix = {prefix + k: v for k, v in state_dict_with_prefix.items()}

        # Atualiza o destino se fornecido, caso contrário, retorna o dicionário com prefixos
        if destination is None:
            destination = state_dict_with_prefix
        else:
            destination.update(state_dict_with_prefix)

        return destination
    # --- Fim da modificação ---


    @abstractmethod
    def initialize_weights(self):
        pass

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
            raise RuntimeError(f"Cannot create layer for {self.prefix}: Original module is None.")

        device = self.orig_module.weight.device
        match self.orig_module:
            case nn.Linear():
                in_features = self.orig_module.in_features
                out_features = self.orig_module.out_features
                # Certifica-se que self.rank existe
                if not hasattr(self, 'rank'): raise AttributeError(f"Rank not set for {self.prefix}")
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
                if not hasattr(self, 'rank'): raise AttributeError(f"Rank not set for {self.prefix}")
                lora_down = Conv2d(in_channels, self.rank, kernel_size, stride, padding, dilation=dilation, bias=False, device=device)
                lora_up = Conv2d(self.rank, out_channels // groups, (1, 1), stride=1, padding=0, bias=False, device=device) # Stride e padding explícitos


            case _:
                raise NotImplementedError(f"Layer creation not implemented for type: {type(self.orig_module)}")

        return lora_down, lora_up

    @classmethod
    def make_dummy(cls):
        """Create a dummy version of a PEFT class."""
        class Dummy(cls):
            def __init__(self, *args, **kwargs):
                prefix = args[0] if args else kwargs.get('prefix', 'dummy_prefix')
                PeftBase.__init__(self, prefix, None) # Chama __init__ da base sem módulo real
                self._state_dict = {}
                self._initialized = False # Dummies não são 'inicializados' no sentido de ter pesos

            def forward(self, *args, **kwargs):
                 raise NotImplementedError("Dummy module should not perform forward pass.")

            def load_state_dict(self, state_dict: Mapping[str, Any],
                                strict: bool = True, assign: bool = False):
                module_prefix_with_dot = self.prefix
                relevant_keys = {k: v for k, v in state_dict.items() if k.startswith(module_prefix_with_dot)}
                state_dict_local = {k.removeprefix(module_prefix_with_dot): v for k, v in relevant_keys.items()}

                if not state_dict_local:
                    return nn.modules.module._IncompatibleKeys([], [])

                self._initialized = True # Marca como 'inicializado' para ter um state_dict
                self._state_dict = copy.deepcopy(state_dict_local)

                keys_to_remove = list(relevant_keys.keys())
                for key in keys_to_remove:
                     if key in state_dict:
                         state_dict.pop(key)

                return nn.modules.module._IncompatibleKeys([], [])

            def state_dict(self, *args, destination=None, prefix='', keep_vars=False): # Assinatura completa
                if not self._initialized:
                    # Retorna vazio se nunca carregou nada
                    if destination is None:
                         destination = {}
                    return destination

                # Adiciona o prefixo do módulo (self.prefix)
                state_dict_with_module_prefix = {self.prefix + k: v for k, v in self._state_dict.items()}

                # Adiciona o prefixo externo (geralmente da chamada pai)
                if prefix:
                    state_dict_with_full_prefix = {prefix + k: v for k, v in state_dict_with_module_prefix.items()}
                else:
                    state_dict_with_full_prefix = state_dict_with_module_prefix


                if destination is None:
                    destination = state_dict_with_full_prefix
                else:
                    destination.update(state_dict_with_full_prefix)

                return destination

            def initialize_weights(self): pass # Não faz nada
            def hook_to_module(self): raise NotImplementedError("Should never be called on a dummy module.")
            def remove_hook_from_module(self): raise NotImplementedError("Should never be called on a dummy module.")
            def apply_to_module(self): raise NotImplementedError("Should never be called on a dummy module.")
            def extract_from_module(self, base_module: nn.Module): raise NotImplementedError("Should never be called on a dummy module.")

        return Dummy


class LoHaModule(PeftBase):
    """Implementation of LoHa from Lycoris."""
    rank: int
    dropout: Dropout
    hada_w1_a: Parameter | None
    hada_w1_b: Parameter | None
    hada_w2_a: Parameter | None
    hada_w2_b: Parameter | None

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super().__init__(prefix, orig_module)
        self.rank = rank
        self.dropout = Dropout(0)
        self.register_buffer("alpha", torch.tensor(alpha))
        self.hada_w1_a = None
        self.hada_w1_b = None
        self.hada_w2_a = None
        self.hada_w2_b = None

        if orig_module is not None:
             try: # Tenta inicializar, mas permite falha se o tipo de camada não for suportado
                self.initialize_weights()
                self.alpha = self.alpha.to(orig_module.weight.device)
             except NotImplementedError:
                 print(f"Aviso: Não foi possível inicializar LoHa para {prefix} devido ao tipo de camada.")
        #self.alpha.requires_grad_(False)

    def initialize_weights(self):
        if self._initialized: return
        self._initialized = True
        # Cria camadas temporárias para obter os pesos Parameter corretos
        hada_w1_b_layer, hada_w1_a_layer = self.create_layer()
        hada_w2_b_layer, hada_w2_a_layer = self.create_layer()

        # Atribui os pesos como Parameter para registro correto
        self.hada_w1_a = hada_w1_a_layer.weight
        self.hada_w1_b = hada_w1_b_layer.weight
        self.hada_w2_a = hada_w2_a_layer.weight
        self.hada_w2_b = hada_w2_b_layer.weight

        # Inicialização
        nn.init.normal_(self.hada_w1_a, std=0.1)
        nn.init.normal_(self.hada_w1_b, std=1)
        nn.init.constant_(self.hada_w2_a, 0)
        nn.init.normal_(self.hada_w2_b, std=1)

    def check_initialized(self):
        super().check_initialized()
        assert self.hada_w1_a is not None
        assert self.hada_w1_b is not None
        assert self.hada_w2_a is not None
        assert self.hada_w2_b is not None

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        if self.op is None: # Se a camada original não era suportada
             print(f"Aviso: Pulando forward de LoHa para {self.prefix} (tipo de camada não suportado). Retornando saída original.")
             return self.orig_forward(x)

        W1 = self.make_weight(self.dropout(self.hada_w1_b),
                              self.dropout(self.hada_w1_a))
        W2 = self.make_weight(self.dropout(self.hada_w2_b),
                              self.dropout(self.hada_w2_a))
        W = (W1 * W2) * (self.alpha.item() / self.rank) # Usa .item()
        return self.orig_forward(x) + self.op(x, W, bias=None, **self.layer_kwargs)

    def apply_to_module(self): pass # TODO
    def extract_from_module(self, base_module: nn.Module): pass # TODO


class LoRAModule(PeftBase):
    lora_down: nn.Module | None
    lora_up: nn.Module | None
    rank: int
    alpha: torch.Tensor
    dropout: Dropout

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super().__init__(prefix, orig_module)
        self.rank = rank
        self.dropout = Dropout(0)
        self.register_buffer("alpha", torch.tensor(alpha))
        self.lora_down = None
        self.lora_up = None

        if orig_module is not None:
            try: # Tenta inicializar, mas permite falha se o tipo de camada não for suportado
                self.initialize_weights()
                self.alpha = self.alpha.to(orig_module.weight.device)
            except NotImplementedError:
                 print(f"Aviso: Não foi possível inicializar LoRA para {prefix} devido ao tipo de camada.")
        # self.alpha.requires_grad_(False)

    def initialize_weights(self):
        if self._initialized: return
        self._initialized = True
        self.lora_down, self.lora_up = self.create_layer()
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def check_initialized(self):
        super().check_initialized()
        assert self.lora_down is not None, f"LoRA down not initialized for {self.prefix}"
        assert self.lora_up is not None, f"LoRA up not initialized for {self.prefix}"

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        if self.op is None: # Se a camada original não era suportada
             print(f"Aviso: Pulando forward de LoRA para {self.prefix} (tipo de camada não suportado). Retornando saída original.")
             return self.orig_forward(x)

        lora_output = self.lora_up(self.lora_down(self.dropout(x)))
        original_output = self.orig_forward(x)
        scale = self.alpha.item() / self.rank
        return original_output + lora_output * scale

    def apply_to_module(self): pass # TODO
    def extract_from_module(self, base_module: nn.Module): pass # TODO


class DoRAModule(LoRAModule):
    """Weight-decomposed low rank adaptation."""
    dora_num_dims: int
    dora_scale: Parameter | None
    norm_epsilon: bool

    def __init__(self, *args, **kwargs):
        self.dora_scale = None
        self.norm_epsilon = kwargs.pop('norm_epsilon', False)
        self.train_device = kwargs.pop('train_device')
        super().__init__(*args, **kwargs) # Chama __init__ de LoRAModule

    def initialize_weights(self):
        if self._initialized: return
        # Inicializa lora_down e lora_up primeiro
        # A chamada a super() pode falhar se o tipo de camada não for suportado
        try:
            super().initialize_weights()
        except NotImplementedError:
             # Se LoRAModule.initialize_weights falhar, não podemos continuar
             self._initialized = False # Garante que não seja marcado como inicializado
             raise # Re-levanta a exceção original

        # Inicializa dora_scale SE a inicialização da base funcionou
        if self.orig_module is not None and self.dora_scale is None:
             orig_weight = get_unquantized_weight(self.orig_module, torch.float, self.train_device)
             self.dora_num_dims = orig_weight.dim() - 1
             initial_norm = torch.norm(
                 orig_weight.transpose(1, 0).reshape(orig_weight.shape[1], -1),
                 dim=1, keepdim=True
             ).reshape(orig_weight.shape[1], *[1] * self.dora_num_dims).transpose(1, 0)
             self.dora_scale = nn.Parameter(initial_norm.to(device=self.orig_module.weight.device))
             del orig_weight
        # Marca como inicializado apenas se tudo correu bem
        # (super já fez self._initialized = True se chegou até aqui sem erro)

    def check_initialized(self):
        super().check_initialized()
        assert self.dora_scale is not None, f"DoRA scale not initialized for {self.prefix}"

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        if self.op is None: # Se a camada original não era suportada
             print(f"Aviso: Pulando forward de DoRA para {self.prefix} (tipo de camada não suportado). Retornando saída original.")
             return self.orig_forward(x)

        A = self.lora_down.weight
        B = self.lora_up.weight
        orig_weight = get_unquantized_weight(self.orig_module, A.dtype, self.train_device)
        lora_delta_w = self.make_weight(A, B) * (self.alpha.item() / self.rank)
        WP = orig_weight + lora_delta_w
        del orig_weight, lora_delta_w

        eps = torch.finfo(WP.dtype).eps if self.norm_epsilon else 0.0
        norm = WP.detach() \
                 .transpose(0, 1) \
                 .reshape(WP.shape[1], -1) \
                 .norm(dim=1, keepdim=True) \
                 .reshape(WP.shape[1], *[1] * self.dora_num_dims) \
                 .transpose(0, 1) + eps
        WP_normalized = WP / norm
        WP_scaled = self.dora_scale * WP_normalized
        x_dropout = self.dropout(x)
        return self.op(x_dropout,
                       WP_scaled,
                       self.orig_module.bias,
                       **self.layer_kwargs)


# Cria instâncias Dummy após as definições das classes originais
DummyLoRAModule = LoRAModule.make_dummy()
DummyDoRAModule = DoRAModule.make_dummy()
DummyLoHaModule = LoHaModule.make_dummy()


class LoRAModuleWrapper:
    orig_module: nn.Module
    rank: int
    alpha: float
    lora_modules: dict[str, PeftBase] # Dicionário para armazenar módulos LoRA/DoRA/Dummy

    def __init__(
            self,
            orig_module: nn.Module | None,
            prefix: str,
            config: TrainConfig,
            module_filter: list[str] = None,
    ):
        self.orig_module = orig_module
        self.prefix = prefix # Prefixo base, ex: "lora_unet"
        self.peft_type = config.peft_type
        self.rank = config.lora_rank
        self.alpha = config.lora_alpha
        self.module_filter = [x.strip() for x in module_filter] if module_filter is not None else []
        weight_decompose = config.lora_decompose

        self.additional_args = [self.rank, self.alpha]
        self.additional_kwargs = {}

        if self.peft_type == PeftType.LORA:
            if weight_decompose:
                self.klass = DoRAModule
                self.dummy_klass = DummyDoRAModule
                self.additional_kwargs = {
                    'norm_epsilon': config.lora_decompose_norm_epsilon,
                    'train_device': torch.device(config.train_device),
                }
            else:
                self.klass = LoRAModule
                self.dummy_klass = DummyLoRAModule
        elif self.peft_type == PeftType.LOHA:
            self.klass = LoHaModule
            self.dummy_klass = DummyLoHaModule
        else:
             raise ValueError(f"Unsupported PeftType: {self.peft_type}")

        self.lora_modules = self.__create_modules(orig_module)

        # --- Geração automática do arquivo de chaves ---
        # Nota: Considere tornar isso condicional com base em um parâmetro de configuração
        # if config.get('generate_lbw_file_on_init', False): # Exemplo de condição
        self.generate_keys_by_block_file(f"{self.prefix}_keys_by_block.txt")
        # <--- ADICIONADO PARA GERAR NA INICIALIZAÇÃO
        # ----------------------------------------------


    def __create_modules(self, root_module: nn.Module | None) -> dict[str, PeftBase]:
        lora_modules = {}
        if root_module is not None:
            for name, child_module in root_module.named_modules():
                # Constrói o prefixo completo para o módulo PEFT
                # Remove o prefixo raiz se name já o contiver (acontece às vezes)
                clean_name = name.removeprefix(self.prefix + "_") if name.startswith(self.prefix + "_") else name
                full_peft_prefix = f"{self.prefix}_{clean_name}" if clean_name else self.prefix

                # Verifica o filtro do módulo
                passes_filter = not self.module_filter or any(f in name for f in self.module_filter)

                if passes_filter and isinstance(child_module, (Linear, Conv2d)):
                    # Cria a instância do módulo PEFT (LoRA, DoRA, LoHa)
                    try:
                        lora_modules[name] = self.klass(full_peft_prefix, child_module, *self.additional_args, **self.additional_kwargs)
                        # print(f"Created {self.klass.__name__} for: {full_peft_prefix} (orig name: {name})") # Debug
                    except Exception as e:
                         print(f"Erro ao criar módulo PEFT para {name} (prefixo {full_peft_prefix}): {e}")


        return lora_modules

    def requires_grad_(self, requires_grad: bool):
        for module in self.lora_modules.values():
            if not isinstance(module, (DummyLoRAModule, DummyDoRAModule, DummyLoHaModule)):
                 module.requires_grad_(requires_grad)

    def parameters(self) -> list[Parameter]:
        parameters = []
        for module in self.lora_modules.values():
            if not isinstance(module, (DummyLoRAModule, DummyDoRAModule, DummyLoHaModule)):
                parameters.extend(list(module.parameters())) # Garante que seja uma lista
        return parameters

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> 'LoRAModuleWrapper':
        for module in self.lora_modules.values():
            module.to(device, dtype)
        return self

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        """Loads the state dict into LoRA/DoRA modules and handles remaining keys with dummies."""
        remaining_state_dict = copy.deepcopy(state_dict)
        loaded_prefixes = set()

        for name, module in self.lora_modules.items():
             # Verifica se o módulo é real antes de tentar carregar
             if not isinstance(module, (DummyLoRAModule, DummyDoRAModule, DummyLoHaModule)):
                 module.load_state_dict(remaining_state_dict)
                 loaded_prefixes.add(module.prefix)


        potential_dummy_keys = {k: v for k, v in remaining_state_dict.items() if k.startswith(self.prefix)}

        while potential_dummy_keys:
            key_example = next(iter(potential_dummy_keys))
            parts = key_example.split('.')
            dummy_prefix_parts = []
            known_suffix = False

            # Verifica terminações comuns para determinar onde cortar o prefixo
            common_suffixes = [
                '.lora_down.weight', '.lora_up.weight', '.alpha', '.dora_scale',
                '.hada_w1_a', '.hada_w1_b', '.hada_w2_a', '.hada_w2_b'
                # Adicione outros sufixos de parâmetros PEFT aqui se necessário
            ]
            for suffix in common_suffixes:
                if key_example.endswith(suffix):
                     # Remove o sufixo e o nome do parâmetro final
                     num_parts_to_remove = suffix.count('.') + 1
                     dummy_prefix_parts = parts[:-num_parts_to_remove]
                     known_suffix = True
                     break

            if not known_suffix:
                 print(f"Aviso: Não foi possível determinar o prefixo do módulo para a chave remanescente: {key_example}. Pulando.")
                 potential_dummy_keys.pop(key_example)
                 continue

            dummy_module_prefix_with_dot = '.'.join(dummy_prefix_parts) + '.'

            if dummy_module_prefix_with_dot in loaded_prefixes:
                 keys_to_remove = [k for k in potential_dummy_keys if k.startswith(dummy_module_prefix_with_dot)]
                 for k in keys_to_remove:
                     if k in potential_dummy_keys: potential_dummy_keys.pop(k)
                 continue

            # O nome do módulo 'name' é a parte após self.prefix_ (ou vazio se for o próprio prefix)
            dummy_module_name = dummy_module_prefix_with_dot.removeprefix(self.prefix + "_").removesuffix('.') if dummy_module_prefix_with_dot != self.prefix + "." else ""
            # print(f"Creating dummy for prefix: {dummy_module_prefix_with_dot} (name: {dummy_module_name})") # Debug
            dummy_module = self.dummy_klass(dummy_module_prefix_with_dot, None, *self.additional_args, **self.additional_kwargs)
            # Passa uma cópia para o dummy processar, preservando potential_dummy_keys para outros dummies
            dummy_module.load_state_dict(copy.deepcopy(potential_dummy_keys))

            # Remove as chaves que o dummy carregou do dict principal
            dummy_keys_loaded = list(dummy_module.state_dict().keys())
            for k in dummy_keys_loaded:
                 if k in potential_dummy_keys:
                     potential_dummy_keys.pop(k)


            self.lora_modules[dummy_module_name] = dummy_module
            loaded_prefixes.add(dummy_module_prefix_with_dot)


        final_remaining_keys = list(potential_dummy_keys.keys())
        if final_remaining_keys:
            #print(f"Warning: Unused keys found after loading state dict for prefix '{self.prefix}': {final_remaining_keys}")
            pass


    def state_dict(self) -> dict:
        """Returns the state dict containing keys from all managed modules (real and dummy)."""
        state_dict = {}
        for module in self.lora_modules.values():
            # state_dict de PeftBase/Dummy já adiciona o prefixo do módulo
            module_sd = module.state_dict()
            state_dict.update(module_sd)
        return state_dict

    def modules(self) -> list[nn.Module]:
        """Returns a list of all managed modules (real and dummy)."""
        return list(self.lora_modules.values())

    def hook_to_module(self):
        """Hooks the real LoRA modules into their corresponding original modules."""
        for module in self.lora_modules.values():
            if not isinstance(module, (DummyLoRAModule, DummyDoRAModule, DummyLoHaModule)):
                module.hook_to_module()

    def remove_hook_from_module(self):
        """Removes the LoRA hook from the real modules."""
        for module in self.lora_modules.values():
             if not isinstance(module, (DummyLoRAModule, DummyDoRAModule, DummyLoHaModule)):
                module.remove_hook_from_module()

    def apply_to_module(self):
        """Applies the LoRA weights directly to the original modules (real modules only)."""
        for module in self.lora_modules.values():
            if not isinstance(module, (DummyLoRAModule, DummyDoRAModule, DummyLoHaModule)):
                module.apply_to_module()

    def extract_from_module(self, base_module: nn.Module):
        """Extracts LoRA weights by comparing the current module state to a base module (real modules only)."""
        for name, module in self.lora_modules.items():
             if not isinstance(module, (DummyLoRAModule, DummyDoRAModule, DummyLoHaModule)):
                try:
                    corresponding_base_submodule = base_module.get_submodule(name)
                    module.extract_from_module(corresponding_base_submodule)
                except AttributeError:
                     print(f"Warning: Could not find corresponding submodule '{name}' in base_module during extraction.")

    def prune(self):
        """Removes all dummy modules from the managed modules."""
        dummy_to_remove = self.dummy_klass
        self.lora_modules = {k: v for (k, v) in self.lora_modules.items() if not isinstance(v, dummy_to_remove)}
        print(f"Pruned dummy modules. Remaining modules: {len(self.lora_modules)}")

    def set_dropout(self, dropout_probability: float):
        """Sets the dropout probability for all real managed modules."""
        if not 0 <= dropout_probability <= 1:
            raise ValueError("Dropout probability must be in [0, 1]")
        for module in self.lora_modules.values():
             if not isinstance(module, (DummyLoRAModule, DummyDoRAModule, DummyLoHaModule)):
                 if hasattr(module, 'dropout') and isinstance(module.dropout, nn.Dropout):
                     module.dropout.p = dropout_probability

    # --- Adicionado para gerar arquivo de chaves por bloco ---
    @staticmethod
    def get_block_id_from_key_prefix(lora_module_prefix_with_dot: str) -> str:
        """Determina o Block ID (IN00-OUT11) a partir do prefixo completo do módulo LoRA/DoRA."""
        lora_module_prefix = lora_module_prefix_with_dot.removesuffix('.')

        # Tratamento especial para mid_block que pode conter subpartes
        if lora_module_prefix.startswith("lora_unet_mid_block"):
            return "M00"

        # Ordena as chaves do mapeamento pela ordem decrescente de comprimento
        # para garantir que os prefixos mais específicos sejam verificados primeiro.
        # Ex: 'lora_unet_down_blocks_0_resnets_0' antes de 'lora_unet_down_blocks_0'
        sorted_map_prefixes = sorted(KEY_TO_BLOCK_MAPPING.keys(), key=len, reverse=True)

        for map_prefix in sorted_map_prefixes:
            if lora_module_prefix.startswith(map_prefix):
                return KEY_TO_BLOCK_MAPPING[map_prefix]

        print(f"Aviso: Prefixo do módulo não mapeado encontrado: {lora_module_prefix}. Assumindo BASE.")
        return "BASE"

    def generate_keys_by_block_file(self, output_filename="unetKeysByBlock_generated.txt"):
        """Gera um arquivo de texto com todas as chaves LoRA/DoRA organizadas por bloco."""
        # Verifica se há módulos para processar
        if not self.lora_modules:
            print("Aviso: Nenhum módulo LoRA/DoRA encontrado no wrapper. Arquivo de chaves não gerado.")
            return

        print(f"Gerando arquivo de chaves por bloco: {output_filename}")
        blocks_data: dict[str, list[str]] = {block_id: [] for block_id in BLOCKID26}

        # Itera sobre os módulos gerenciados (reais e dummies)
        for module_name, module_instance in self.lora_modules.items():
            # Pega o prefixo correto armazenado na instância do módulo PEFT
            module_prefix_with_dot = module_instance.prefix
            block_id = self.get_block_id_from_key_prefix(module_prefix_with_dot)

            # Pega as chaves deste módulo específico usando seu state_dict()
            # que já deve incluir o prefixo correto (module_prefix_with_dot)
            try:
                 module_state_dict = module_instance.state_dict()
                 for key in module_state_dict.keys():
                     blocks_data[block_id].append(key)
            except Exception as e:
                 print(f"Erro ao obter state_dict para o módulo {module_name} (prefixo {module_prefix_with_dot}): {e}")


        # Ordena as chaves dentro de cada bloco
        for block_id in blocks_data:
            blocks_data[block_id].sort()

        # Formata e escreve o arquivo
        output_lines = []
        for block_id in BLOCKID26:
            output_lines.append(f"=== {block_id} ===")
            keys_in_block = blocks_data[block_id]
            if keys_in_block:
                output_lines.extend(keys_in_block)
            else:
                if block_id == "BASE":
                    output_lines.append("(Nenhuma chave UNet corresponde a BASE)")
                elif block_id in ["IN10", "IN11", "OUT09", "OUT10"]:
                     output_lines.append("(Vazio)")
                else:
                    output_lines.append("(Vazio)")

            output_lines.append("") # Linha em branco entre blocos

        try:
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines).strip())
            print(f"Arquivo '{output_filename}' gerado com sucesso.")
        except Exception as e:
            print(f"Erro ao escrever o arquivo '{output_filename}': {e}")
    # --- Fim da adição ---

# --- FIM DO ARQUIVO ---