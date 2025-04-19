import os
import re
import copy
import math
import fnmatch
import traceback

from typing import Any
from datetime import datetime
from abc import abstractmethod
from collections.abc import Mapping
from modules.util.enum.ModelType import PeftType
from modules.util.config.TrainConfig import TrainConfig
from modules.util.quantization_util import get_unquantized_weight, get_weight_shape

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Conv2d, Dropout, Linear, Parameter


class RuleSet:
    """Helper class to manage configuration rules based on layer name patterns."""

    def __init__(self, pattern_dict: dict[str, dict[str, int | float]] | None):
        self.patterns = pattern_dict or {}
        self._sorted_patterns = sorted(self.patterns.keys(), key=len, reverse=True)

    def match(self, name: str) -> dict[str, int | float] | None:
        """Finds the first pattern (most specific to least specific) that matches the name and returns its configuration."""
        for pattern in self._sorted_patterns:
            if fnmatch.fnmatch(name, pattern):
                return self.patterns[pattern]
        return None  # Return None if no pattern matches


BLOCK_IDS = [
    "BASE",   # tudo o que não for UNet (Text Encoders, etc)
    "IN00",   # input conv
    "IN01",   # down_blocks_0
    "IN02",   # down_blocks_1
    "IN03",   # down_blocks_2
    "IN04",   # down_blocks_3
    "M00",    # mid_block
    "OUT04",  # up_blocks_3
    "OUT03",  # up_blocks_2
    "OUT02",  # up_blocks_1
    "OUT01",  # up_blocks_0
    "OUT00",  # conv_out / saída
]

KEY_TO_BLOCK_MAPPING = {
    "lora_unet_conv_in":                    "IN00",
    "lora_unet_time_embedding":             "IN00",

    "lora_unet_down_blocks_0_resnets_0":    "IN01",
    "lora_unet_down_blocks_0_resnets_1":    "IN01",
    "lora_unet_down_blocks_0_downsamplers_0":"IN01",

    "lora_unet_down_blocks_1_resnets_0":    "IN02",
    "lora_unet_down_blocks_1_attentions_0": "IN02",
    "lora_unet_down_blocks_1_resnets_1":    "IN02",
    "lora_unet_down_blocks_1_attentions_1": "IN02",
    "lora_unet_down_blocks_1_downsamplers_0":"IN02",

    "lora_unet_down_blocks_2_resnets_0":    "IN03",
    "lora_unet_down_blocks_2_attentions_0": "IN03",
    "lora_unet_down_blocks_2_resnets_1":    "IN03",
    "lora_unet_down_blocks_2_attentions_1": "IN03",
    "lora_unet_down_blocks_2_downsamplers_0":"IN03",

    "lora_unet_down_blocks_3_resnets_0":    "IN04",
    "lora_unet_down_blocks_3_resnets_1":    "IN04",

    "lora_unet_mid_block":                  "M00",

    "lora_unet_up_blocks_0_resnets_0":      "OUT01",
    "lora_unet_up_blocks_0_attentions_0":   "OUT01",
    "lora_unet_up_blocks_0_resnets_1":      "OUT01",
    "lora_unet_up_blocks_0_attentions_1":   "OUT01",
    "lora_unet_up_blocks_0_resnets_2":      "OUT01",
    "lora_unet_up_blocks_0_attentions_2":   "OUT01",
    "lora_unet_up_blocks_0_upsamplers_0":   "OUT01",

    "lora_unet_up_blocks_1_resnets_0":      "OUT02",
    "lora_unet_up_blocks_1_attentions_0":   "OUT02",
    "lora_unet_up_blocks_1_resnets_1":      "OUT02",
    "lora_unet_up_blocks_1_attentions_1":   "OUT02",
    "lora_unet_up_blocks_1_resnets_2":      "OUT02",
    "lora_unet_up_blocks_1_attentions_2":   "OUT02",
    "lora_unet_up_blocks_1_upsamplers_0":   "OUT02",

    "lora_unet_up_blocks_2_resnets_0":      "OUT03",
    "lora_unet_up_blocks_2_resnets_1":      "OUT03",
    "lora_unet_up_blocks_2_resnets_2":      "OUT03",
    "lora_unet_up_blocks_2_upsamplers_0":   "OUT03",

    "lora_unet_up_blocks_3_resnets_0":      "OUT04",
    "lora_unet_up_blocks_3_resnets_1":      "OUT04",

    "lora_unet_conv_out":                   "OUT00",

    # Text Encoders e afins
    "lora_te1_":                            "BASE",
    "lora_te2_":                            "BASE",
}


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
        self.prefix = prefix + '.'
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
                    raise NotImplementedError(
                        f"Only Linear and Conv2d are supported layers. Got {type(orig_module)} for prefix {self.prefix}"
                    )

    def hook_to_module(self):
        if self._orig_module is None:
            return
        if not self.is_applied:
            self.orig_forward = self.orig_module.forward
            self.orig_train = self.orig_module.train
            self.orig_eval = self.orig_module.eval
            self.orig_module.forward = self.forward
            self.orig_module.train = self._wrap_train
            self.orig_module.eval = self._wrap_eval
            self.is_applied = True

    def remove_hook_from_module(self):
        if self._orig_module is None or self.orig_forward is None:
            return
        assert self.orig_forward is not None  # Mantido do original
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.orig_module.train = self.orig_train
            self.orig_module.eval = self.orig_eval
            self.is_applied = False

    def _wrap_train(self, mode=True):
        if self._orig_module is None or self.orig_train is None:
            return
        self.orig_train(mode)
        self.train(mode)

    def _wrap_eval(self):
        if self._orig_module is None or self.orig_eval is None:
            return
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

        assert self._orig_module is not None, f"orig_module is None for {self.prefix}"
        assert self.orig_forward is not None, f"orig_forward is None for {self.prefix}"

    @property
    def orig_module(self) -> nn.Module:
        if self._orig_module is None:
            raise AttributeError(f"Original module not set for PEFT layer with prefix {self.prefix}")
        return self._orig_module[0]

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        module_prefix_with_dot = self.prefix  # self.prefix already ends with '.'
        relevant_keys = {k: v for k, v in state_dict.items() if k.startswith(module_prefix_with_dot)}

        if not relevant_keys:
            return nn.modules.module._IncompatibleKeys([], [])  # No missing, no unexpected

        state_dict_local = {k.removeprefix(module_prefix_with_dot): v for k, v in relevant_keys.items()}

        if not self._initialized and state_dict_local and self._orig_module is not None:
            try:
                self.initialize_weights()  # Call the subclass implementation
                self._initialized = True  # Mark as initialized *after* success
            except NotImplementedError as e:
                print(f"Error initializing weights for {self.prefix}: {e}. Module might not load correctly.")
            except AttributeError as e:
                print(
                    f"Error during weight initialization for {self.prefix} (likely missing rank/alpha): {e}. Module might not load correctly."
                )

        load_result = nn.modules.module._IncompatibleKeys([], [])  # Default empty result
        if self._initialized:
            try:
                load_result = super().load_state_dict(state_dict_local, strict=strict, assign=assign)
            except Exception as e:
                print(f"Error calling super().load_state_dict for {self.prefix}: {e}")
        else:
            missing_keys = list(state_dict_local.keys())
            load_result = nn.modules.module._IncompatibleKeys(missing_keys, [])

        keys_processed_or_skipped = list(relevant_keys.keys())
        for key in keys_processed_or_skipped:
            if key in state_dict:  # Check if the key still exists (it should)
                state_dict.pop(key)  # Remove keys that were handled (or attempted) here

        return load_result

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        local_state_dict = super().state_dict(*args, destination=None, prefix="", keep_vars=keep_vars)

        state_dict_with_prefix = {self.prefix + k: v for k, v in local_state_dict.items()}

        if prefix:
            state_dict_with_prefix = {prefix + k: v for k, v in state_dict_with_prefix.items()}

        if destination is None:
            destination = state_dict_with_prefix
        else:
            destination.update(state_dict_with_prefix)

        return destination

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
        if self._orig_module is None:
            raise RuntimeError(f"Cannot create layer for {self.prefix}: Original module is None.")
        if not hasattr(self, "rank"):
            raise AttributeError(f"Rank not set for PEFT module {self.prefix} before calling create_layer.")

        device = self.orig_module.weight.device
        match self.orig_module:
            case nn.Linear():
                in_features = self.orig_module.in_features
                out_features = self.orig_module.out_features
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
                    out_channels,         # antes era out_channels // groups
                    (1, 1),
                    stride=1,
                    padding=0,
                    bias=False,
                    device=device,
                )

            case _:
                raise NotImplementedError(f"Layer creation not implemented for type: {type(self.orig_module)}")

        return lora_down, lora_up

    @classmethod
    def make_dummy(cls):
        """Create a dummy version of a PEFT class."""

        class Dummy(cls):
            def __init__(self, *args, **kwargs):
                prefix_arg = args[0] if args else kwargs.get("prefix", "dummy_prefix")
                PeftBase.__init__(self, prefix_arg, None)
                self._state_dict = {}
                self._initialized = False
                self.rank = args[2] if len(args) > 2 else kwargs.get("rank", 0)
                self.alpha_val = args[3] if len(args) > 3 else kwargs.get("alpha", 0.0)

                self.dropout = Dropout(0)  # Dummy dropout

            def forward(self, *args, **kwargs):
                raise NotImplementedError("Dummy module should not perform forward pass.")

            def load_state_dict(
                self,
                state_dict: Mapping[str, Any],
                strict: bool = True,
                assign: bool = False,
            ):
                module_prefix_with_dot = self.prefix
                relevant_keys = {k: v for k, v in state_dict.items() if k.startswith(module_prefix_with_dot)}
                state_dict_local = {k.removeprefix(module_prefix_with_dot): v for k, v in relevant_keys.items()}

                if not state_dict_local:
                    return nn.modules.module._IncompatibleKeys([], [])

                self._initialized = True
                self._state_dict = copy.deepcopy(state_dict_local)

                # Prioriza 'alpha' do state_dict, senão usa o valor de init
                if "alpha" in self._state_dict:
                    alpha_tensor = self._state_dict["alpha"]
                    if not isinstance(alpha_tensor, torch.Tensor):
                        # Tenta converter se não for tensor (ex: float carregado de JSON)
                        try:
                            alpha_tensor = torch.tensor(float(alpha_tensor))
                        except ValueError:
                            print(f"Warning: Could not convert alpha value {alpha_tensor} to tensor for dummy {self.prefix}")
                            alpha_tensor = torch.tensor(0.0) # Fallback
                elif self.alpha_val is not None:
                    alpha_tensor = torch.tensor(float(self.alpha_val)) # Garante float
                else:
                    alpha_tensor = torch.tensor(0.0) # Fallback se nem state_dict nem init tiverem alpha

                if not hasattr(self, "alpha"):
                    self.register_buffer("alpha", alpha_tensor.clone())
                else:
                    # Garante que o buffer existente tenha o mesmo device/dtype antes de copiar
                    if self.alpha.device != alpha_tensor.device or self.alpha.dtype != alpha_tensor.dtype:
                         self.alpha = alpha_tensor.clone().to(device=self.alpha.device, dtype=self.alpha.dtype)
                    else:
                        self.alpha.copy_(alpha_tensor)
                self.alpha.requires_grad_(False)

                keys_to_remove = list(relevant_keys.keys())
                for key in keys_to_remove:
                    if key in state_dict:
                        state_dict.pop(key)

                return nn.modules.module._IncompatibleKeys([], [])

            def initialize_weights(self):
                pass  # Does nothing for a dummy

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
            try:
                self.initialize_weights()
                self.alpha = self.alpha.to(orig_module.weight.device)
            except NotImplementedError:
                print(f"Warning: LoHa init failed for {prefix} due to unsupported layer type.")
        if hasattr(self, "alpha"):
            self.alpha.requires_grad_(False)

    def initialize_weights(self):
        if self._initialized:
            return
        if self._orig_module is None:
            return  # Cannot initialize without original module

        hada_w1_b_module, hada_w1_a_module = self.create_layer()
        hada_w2_b_module, hada_w2_a_module = self.create_layer()

        self.hada_w1_a = Parameter(hada_w1_a_module.weight)
        self.hada_w1_b = Parameter(hada_w1_b_module.weight)
        self.hada_w2_a = Parameter(hada_w2_a_module.weight)
        self.hada_w2_b = Parameter(hada_w2_b_module.weight)

        nn.init.normal_(self.hada_w1_a, std=0.1)
        nn.init.normal_(self.hada_w1_b, std=1)
        nn.init.constant_(self.hada_w2_a, 0)
        nn.init.normal_(self.hada_w2_b, std=1)
        self._initialized = True

    def check_initialized(self):
        super().check_initialized()
        assert self.hada_w1_a is not None
        assert self.hada_w1_b is not None
        assert self.hada_w2_a is not None
        assert self.hada_w2_b is not None
        assert hasattr(self, "alpha"), f"Alpha buffer missing in LoHa {self.prefix}"

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        if self.op is None:  # If the original layer was not supported
            print(f"Warning: Skipping LoHa forward for {self.prefix} (unsupported layer type). Returning original output.")
            return self.orig_forward(x)

        W1 = self.make_weight(self.dropout(self.hada_w1_b), self.dropout(self.hada_w1_a))
        W2 = self.make_weight(self.dropout(self.hada_w2_b), self.dropout(self.hada_w2_a))
        scale = self.alpha.item() / self.rank
        W = (W1 * W2) * scale
        return self.orig_forward(x) + self.op(x, W, bias=None, **self.layer_kwargs)

    def apply_to_module(self):
        # TODO: Implement merging logic if needed
        raise NotImplementedError

    def extract_from_module(self, base_module: nn.Module):
        # TODO: Implement extraction logic if needed
        raise NotImplementedError


class LoRAModule(PeftBase):
    lora_down: nn.Module | None
    lora_up: nn.Module | None
    rank: int
    alpha: Parameter | None  # Mudar para Parameter (ou manter Tensor se não for treinável)
    dropout: Dropout

    def __init__(self, prefix: str, orig_module: nn.Module | None, rank: int, alpha: float):
        super().__init__(prefix, orig_module)

        self.rank = rank
        self.dropout = Dropout(0)
        self.alpha = None  # Será inicializado em initialize_weights
        self.alpha_val = alpha  # Guardar valor para inicialização
        self.lora_down = None
        self.lora_up = None

        if orig_module is not None:
            try:
                self.initialize_weights()
            except NotImplementedError:
                print(f"Warning: LoRA init failed for {prefix} due to unsupported layer type.")

    def initialize_weights(self):
        if self._initialized:
            return
        if self._orig_module is None:
            return

        device = self.orig_module.weight.device

        self.lora_down, self.lora_up = self.create_layer()  # create_layer já usa o device correto

        self.alpha = Parameter(torch.tensor(float(self.alpha_val), device=device), requires_grad=False)  # Não treinável por padrão

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self._initialized = True

    def check_initialized(self):
        super().check_initialized()
        assert self.lora_down is not None, f"LoRA down not initialized for {self.prefix}"
        assert self.lora_up is not None, f"LoRA up not initialized for {self.prefix}"
        assert self.alpha is not None, f"Alpha parameter missing in LoRA {self.prefix}"

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        if self.op is None:  # If the original layer was not supported
            print(f"Warning: Skipping LoRA forward for {self.prefix} (unsupported layer type). Returning original output.")
            return self.orig_forward(x)

        lora_output = self.lora_up(self.lora_down(self.dropout(x)))
        original_output = self.orig_forward(x)
        scale = (self.alpha.to(dtype=x.dtype) / self.rank)
        return original_output + lora_output * scale

    def apply_to_module(self):
        # TODO: Implement merging logic if needed
        raise NotImplementedError

    def extract_from_module(self, base_module: nn.Module):
        # TODO: Implement extraction logic if needed
        raise NotImplementedError


class DoRAModule(LoRAModule):
    """Weight-decomposed low rank adaptation."""

    # dora_num_dims is implicitly handled by norm calculation now
    dora_scale: Parameter | None  # Use Parameter for trainable scale
    norm_epsilon: bool
    train_device: torch.device  # Add train_device attribute

    def __init__(self, *args, **kwargs):
        # Pop DoRA specific kwargs before calling super().__init__
        self.norm_epsilon = kwargs.pop("norm_epsilon", 1e-6)  # Default epsilon
        self.train_device = kwargs.pop(
            "train_device", torch.device("cpu")
        )  # Default device
        self.dora_scale = None  # Initialize as None
        # Call LoRAModule's __init__ with remaining args/kwargs
        super().__init__(*args, **kwargs)
        # Note: initialize_weights is called by super() if orig_module exists

    def initialize_weights(self):
        super().initialize_weights()

        orig_weight = get_unquantized_weight(self.orig_module, torch.float, self.train_device)

        # Thanks to KohakuBlueLeaf once again for figuring out the shape
        # wrangling that works for both Linear and Convolutional layers. If you
        # were just doing this for Linear, it would be substantially simpler.
        self.dora_num_dims = orig_weight.dim() - 1
        self.dora_scale = nn.Parameter(
            torch.norm(
                orig_weight.transpose(1, 0).reshape(orig_weight.shape[1], -1),
                dim=1, keepdim=True)
            .reshape(orig_weight.shape[1], *[1] * self.dora_num_dims)
            .transpose(1, 0)
            .to(device=self.orig_module.weight.device)
        )

        del orig_weight

    def check_initialized(self):
        super().check_initialized()
        assert (
            self.dora_scale is not None
        ), f"DoRA scale not initialized for {self.prefix}"

    def forward(self, x, *args, **kwargs):
        self.check_initialized()

        A = self.lora_down.weight
        B = self.lora_up.weight
        orig_weight = get_unquantized_weight(self.orig_module, A.dtype, self.train_device)
        WP = orig_weight + (self.make_weight(A, B) * (self.alpha / self.rank))
        del orig_weight
        # A norm should never really end up zero at any point, but epsilon just
        # to be safe if we underflow or something. Also, as per section 4.3 of
        # the paper, we treat the norm as a constant for the purposes of
        # backpropagation in order to save VRAM (to do this, we detach it from
        # the gradient graph).
        eps = torch.finfo(WP.dtype).eps if self.norm_epsilon else 0.0
        norm = WP.detach() \
                 .transpose(0, 1) \
                 .reshape(WP.shape[1], -1) \
                 .norm(dim=1, keepdim=True) \
                 .reshape(WP.shape[1], *[1] * self.dora_num_dims) \
                 .transpose(0, 1) + eps
        WP = self.dora_scale * (WP / norm)
        # In the DoRA codebase (and thus the paper results), they perform
        # dropout on the *input*, rather than between layers, so we duplicate
        # that here.
        return self.op(self.dropout(x),
                       WP,
                       self.orig_module.bias,
                       **self.layer_kwargs)

DummyLoRAModule = LoRAModule.make_dummy()
DummyDoRAModule = DoRAModule.make_dummy()
DummyLoHaModule = LoHaModule.make_dummy()


class LoRAModuleWrapper:
    """
    Manages PEFT modules (LoRA, LoHa, DoRA) for a PyTorch module.

    Applies a global rank and alpha to selected layers based on filters.
    Supports `module_filter` (via external_module_filter or presets) and `lora_layers_blacklist`.
    """

    orig_module: nn.Module | None
    prefix: str
    peft_type: PeftType
    config: TrainConfig
    default_rank: int
    default_alpha: float
    inclusion_filter_patterns: list[str]
    exclusion_filter_patterns: list[str]
    klass: type[PeftBase]
    dummy_klass: type[PeftBase]
    global_additional_kwargs: dict
    lora_modules: dict[str, PeftBase]

    def __init__(
        self,
        orig_module: nn.Module | None,
        prefix: str,
        config: TrainConfig,
        external_module_filter: list[str] | None = None,
    ):
        """
        Initializes the Wrapper.

        Args:
            orig_module: The original nn.Module to be adapted (can be None).
            prefix: Prefix added to PEFT module keys (e.g., "lora_unet").
            config: TrainConfig object containing settings:
                - peft_type: PeftType (LORA, LOHA, DORA).
                - lora_rank: Global rank to use for all adapted layers.
                - lora_alpha: Global alpha to use for all adapted layers.
                - lora_layers_blacklist: List of patterns to explicitly exclude.
                - lora_decompose: True to use DoRA instead of LoRA.
                - lora_decompose_norm_epsilon: Epsilon for DoRA norm.
                - train_device: Device for DoRA calculations.
            preset_name: Optional name of a preset configuration (e.g., 'attn').
                         Presets define inclusion filters *only*. Rank/alpha are global.
            external_module_filter: Optional list of patterns. If provided, acts as an
                                   *additional* inclusion filter alongside the preset's filter.
        """
        if orig_module is None:
            print(f"Info: LoRAModuleWrapper '{prefix}' initialized without an original module. Will only load from state_dict.")
        self.orig_module = orig_module
        self.prefix = prefix
        self.peft_type = config.peft_type
        self.config = config

        self.default_rank = config.lora_rank
        self.default_alpha = config.lora_alpha
        print(f"[LoRA INFO] Global Rank = {self.default_rank}, Global Alpha = {self.default_alpha}")

        try:
            from modules.modelSetup.StableDiffusionXLLoRASetup import PRESETS
        except ImportError:
            print("Warning: PRESETS dictionary not found. Presets filters will not be loaded.")
            PRESETS = {}

        preset_inclusion_patterns = []
        preset_name = config.lora_layer_preset  # Usa o preset do config

        if preset_name:
            # Usa config.lora_layer_preset
            preset_config_raw = PRESETS.get(preset_name)
            # Adaptação: No seu StableDiffusionXLLoRASetup, PRESETS parece ser List[str]
            if preset_config_raw is None and preset_name != "full":
                print(f"[LoRA WARNING] Preset '{preset_name}' not found or is None. Using external filter/blacklist only.")
            elif preset_name == "full":
                print("[LoRA INFO] Preset 'full' selected. No preset-specific inclusion filter applied.")
            # Verifica se é uma lista de strings (como parece ser no seu setup)
            elif isinstance(preset_config_raw, list):
                print(f"[LoRA INFO] Using preset '{preset_name}' for inclusion filters.")
                # Os itens da lista são os padrões de inclusão
                preset_inclusion_patterns = [p for p in preset_config_raw if isinstance(p, str)]
            else:
                print(
                    f"[LoRA WARNING] Preset '{preset_name}' has unexpected format ({type(preset_config_raw)}). Expected list[str]. Ignoring preset filters."
                )
        else:
            print("[LoRA INFO] No preset specified. Using external filter/blacklist only.")

        # Combina filtros de inclusão (Preset + Externo)
        combined_inclusion_patterns = set(p.strip() for p in preset_inclusion_patterns if p.strip())
        if external_module_filter:
            # Nota: O seu __init__ original passava config.lora_layers como external_module_filter
            # Isso parece confuso. Vamos manter o argumento `external_module_filter` explícito.
            # Se você quer usar `config.lora_layers` aqui, passe-o ao instanciar o Wrapper.
            combined_inclusion_patterns.update(p.strip() for p in external_module_filter if p.strip())

        self.inclusion_filter_patterns = list(combined_inclusion_patterns)
        print(
            f"[LoRA INFO] Active inclusion patterns: {self.inclusion_filter_patterns if self.inclusion_filter_patterns else 'None (include all unless blacklisted)'}"
        )

        # Prepara filtro de exclusão (Blacklist)
        self.exclusion_filter_patterns = [b.strip() for b in (config.lora_layers_blacklist or []) if b.strip()]
        print(
            f"[LoRA INFO] Active exclusion patterns (blacklist): {self.exclusion_filter_patterns if self.exclusion_filter_patterns else 'None'}"
        )

        self.layer_rules = config.lora_layer_rules or {}
        
        print(f"[LoRA INFO self.layer_rules: {self.layer_rules}")

        self._sorted_rule_patterns = sorted(self.layer_rules.keys(), key=len, reverse=True)
        print(f"[LoRA INFO self._sorted_rule_patterns: {self._sorted_rule_patterns}")

        if self.layer_rules:
            print(f"[LoRA INFO] Loaded {len(self.layer_rules)} layer-specific rank/alpha rules:")
            for pattern in self._sorted_rule_patterns:
                print(f"  - Rule: '{pattern}' -> {self.layer_rules[pattern]}")
        else:
            print("[LoRA INFO] No layer-specific rank/alpha rules defined. Using global defaults.")

        self.global_additional_kwargs = {}
        if self.peft_type == PeftType.LORA:
            if config.lora_decompose:
                self.klass = DoRAModule
                self.dummy_klass = DummyDoRAModule
                self.global_additional_kwargs = {
                    "norm_epsilon": config.lora_decompose_norm_epsilon,
                    "train_device": torch.device(config.train_device),
                }
                print(
                    f"[LoRA INFO] Using DoRA (decompose=True) with epsilon={self.global_additional_kwargs['norm_epsilon']}, device={self.global_additional_kwargs['train_device']}"
                )
            else:
                self.klass = LoRAModule
                self.dummy_klass = DummyLoRAModule
                print("[LoRA INFO] Using LoRA (decompose=False)")
        elif self.peft_type == PeftType.LOHA:
            self.klass = LoHaModule
            self.dummy_klass = DummyLoHaModule
            print("[LoRA INFO] Using LoHa")
        else:
            raise ValueError(f"Unsupported PeftType: {self.peft_type}")

        # Cria módulos PEFT (agora usará as regras)
        self.lora_modules = self._initialize_peft_modules(orig_module)
        print(f"[LoRA INFO] LoRAModuleWrapper '{self.prefix}' initialized with {len(self.lora_modules)} PEFT modules.")
        self.generate_keys_by_block_file()

    def _should_include_module(self, original_module_name: str, potential_peft_prefix_with_dot: str) -> bool:
        """Checks if a module should be included based on its original name and potential PEFT prefix,
        considering exclusion and inclusion filters."""
        peft_prefix_no_dot = potential_peft_prefix_with_dot.removesuffix(".")

        for pattern in self.exclusion_filter_patterns:
            contains_pattern = f"*{pattern}*"
            match_orig = fnmatch.fnmatch(original_module_name, contains_pattern)
            match_peft = fnmatch.fnmatch(peft_prefix_no_dot, contains_pattern)

            if match_orig or match_peft:
                return False

        if not self.inclusion_filter_patterns:
            return True
        else:
            matched_inclusion = False
            for pattern in self.inclusion_filter_patterns:
                contains_pattern = f"*{pattern}*"
                match_orig = fnmatch.fnmatch(original_module_name, contains_pattern)
                match_peft = fnmatch.fnmatch(peft_prefix_no_dot, contains_pattern)

                if match_orig or match_peft:
                    matched_inclusion = True
                    break
            if matched_inclusion:
                return True
            else:
                return False

    def _initialize_peft_modules(self, root_module: nn.Module | None) -> dict[str, PeftBase]:
        """Identifica, filtra e cria módulos PEFT usando regras de rank/alpha."""
        lora_modules: dict[str, PeftBase] = {}
        if root_module is None:
            print("[LoRA WARNING] _initialize_peft_modules called without root_module. No PEFT modules created.")
            return lora_modules

        print("[LoRA INFO] Identifying and creating PEFT modules...")
        modules_created_count = 0
        modules_skipped_type = 0
        modules_skipped_filter = 0

        for name, child_module in root_module.named_modules():
            # Pula tipos não suportados
            if not isinstance(child_module, (Linear, Conv2d)):
                # Conta como skip de tipo APENAS se não for Linear/Conv2d
                modules_skipped_type += 1
                continue

            # Gera prefixo PEFT potencial
            original_layer_name = name # Mantém o nome original para usar como chave no dict e logs
            peft_compatible_name_part = original_layer_name.replace('.', '_') # Converte '.' para '_' para o prefixo PEFT
            potential_peft_prefix_argument = f"{self.prefix}_{peft_compatible_name_part}" # Ex: "lora_unet_down_blocks_0_attn1_to_q"
            potential_peft_prefix_with_dot_for_filter = potential_peft_prefix_argument + "."

            # Verifica se deve incluir baseado nos filtros
            should_include = self._should_include_module(original_layer_name, potential_peft_prefix_with_dot_for_filter)

            if should_include:
                peft_module_prefix_for_constructor = potential_peft_prefix_argument

                # --- NOVO: Determina Rank/Alpha usando Regras ---
                rank_to_use = self.default_rank
                alpha_to_use = self.default_alpha
                rule_applied = "Global Default"

                # Verifica regras contra nome original e prefixo PEFT
                target_names_for_rules = [original_layer_name, peft_module_prefix_for_constructor]
                matched_rule = False
                for pattern in self._sorted_rule_patterns:  # Itera sobre regras ordenadas
                    for target_name in target_names_for_rules:
                        # Usa fnmatch padrão (ou `f"*{pattern}*"` se preferir 'contains')
                        if fnmatch.fnmatch(target_name, pattern):
                            rule_config = self.layer_rules[pattern]
                            # Pega rank/alpha da regra, fallback para default global
                            rank_to_use = rule_config.get("rank", self.default_rank)
                            alpha_to_use = rule_config.get("alpha", self.default_alpha)
                            rule_applied = f"Rule ('{pattern}')"
                            matched_rule = True
                            break  # Para de checar alvos para este padrão
                    if matched_rule:
                        break  # Para de checar padrões (primeiro match vence)
                # --- FIM NOVO ---

                # Prepara args/kwargs para o construtor do PEFT
                args_for_this_module = [
                    peft_module_prefix_for_constructor,
                    child_module,
                    rank_to_use,  # Usa rank determinado
                    alpha_to_use,  # Usa alpha determinado
                ]
                kwargs_for_this_module = self.global_additional_kwargs.copy()

                # Tenta criar o módulo PEFT
                try:
                    # Atualiza log para mostrar rank/alpha aplicados e a regra
                    log_msg = (
                    f"[LoRA CREATE] {self.klass.__name__} for: '{original_layer_name}' " # Loga o nome original
                    f"({rule_applied}: Rank={rank_to_use}, Alpha={alpha_to_use}) "
                    f"| PEFT Prefix: {peft_module_prefix_for_constructor}" # Loga o prefixo PEFT final
                )
                #print(log_msg)  # Mantém o log para feedback

                    lora_modules[original_layer_name] = self.klass(*args_for_this_module, **kwargs_for_this_module)
                    modules_created_count += 1

                except Exception as e:
                    print(
                        f"[LoRA ERROR] Failed to create PEFT module for layer '{name}' (prefix {peft_module_prefix_for_constructor}) "
                        f"using Rank={rank_to_use}, Alpha={alpha_to_use}: {e}"
                    )

            else:
                # Conta como skip de filtro SOMENTE se for Linear/Conv2d mas foi filtrado
                modules_skipped_filter += 1

        # Resumo final
        print(f"[LoRA INFO] Module Creation Summary for '{self.prefix}':")
        print(f"  - Created: {modules_created_count} PEFT modules.")
        print(f"  - Skipped (Filter): {modules_skipped_filter} Linear/Conv2d modules.")
        print(f"  - Skipped (Type): {modules_skipped_type} non-Linear/Conv2d modules.")
        return lora_modules

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        """
        Loads the state dict into managed PEFT modules (real and dummy).

        Handles creating dummy modules for keys present in the state_dict
        but not corresponding to any initially created real PEFT modules.
        Uses the global default rank/alpha when creating dummies if needed.
        """
        if not self.lora_modules and self.orig_module is None:
            print(
                "[LoRA WARNING] load_state_dict called with no original module and no pre-existing PEFT modules. Attempting to load all as dummies."
            )

        remaining_state_dict = copy.deepcopy(state_dict)
        loaded_module_prefixes = set()
        missing_keys_overall = []
        unexpected_keys_overall = [k for k in remaining_state_dict if k.startswith(self.prefix)]

        current_real_module_names = [name for name, mod in self.lora_modules.items() if not isinstance(mod, self.dummy_klass)]

        # Carrega módulos reais primeiro
        for name in current_real_module_names:
            module = self.lora_modules.get(name)
            if module is None:
                continue
            try:
                # Passa o remaining_state_dict para que PeftBase.load_state_dict possa remover as chaves
                result = module.load_state_dict(remaining_state_dict, strict=True)
                missing_keys_overall.extend([module.prefix + k for k in result.missing_keys])
                # Module keys são removidos de remaining_state_dict dentro de PeftBase.load_state_dict
                loaded_module_prefixes.add(module.prefix)
                # Remove chaves carregadas da lista de inesperadas
                module_keys_in_sd = {k for k in state_dict if k.startswith(module.prefix)}
                unexpected_keys_overall = [k for k in unexpected_keys_overall if k not in module_keys_in_sd]
            except Exception as e:
                print(f"Error loading state_dict into real module {name} (prefix {module.prefix}): {e}")

        # Processa chaves restantes para criar dummies
        potential_dummy_keys = {k: v for k, v in remaining_state_dict.items() if k.startswith(self.prefix)}
        processed_dummy_prefixes = set()
        keys_by_prefix = {}
        suffixes_to_strip = [
            "lora_down.weight",
            "lora_up.weight",
            "alpha",
            "dora_scale",
            "hada_w1_a",
            "hada_w1_b",
            "hada_w2_a",
            "hada_w2_b",
        ]

        # Agrupa chaves por prefixo inferido
        for key in list(potential_dummy_keys.keys()):
            inferred_prefix = key
            for suffix in suffixes_to_strip:
                if key.endswith("." + suffix):
                    inferred_prefix = key[: -(len(suffix) + 1)] + "."
                    break
            if inferred_prefix not in keys_by_prefix:
                keys_by_prefix[inferred_prefix] = {}
            keys_by_prefix[inferred_prefix][key] = potential_dummy_keys[key]

        # Cria dummies
        for peft_prefix, keys_for_this_prefix in keys_by_prefix.items():
            if peft_prefix in loaded_module_prefixes or peft_prefix in processed_dummy_prefixes:
                # Chaves já carregadas ou de dummy já processado, remove de inesperadas
                unexpected_keys_overall = [k for k in unexpected_keys_overall if not k.startswith(peft_prefix)]
                continue

            # Infére nome original (melhor esforço)
            relative_name_part = peft_prefix.removeprefix(self.prefix + "_").removesuffix(".")
            original_layer_name = relative_name_part.replace("_", ".")

            rank_to_use = self.default_rank
            alpha_to_use = self.default_alpha
            rule_applied = "Global Default"
            matched_rule = False
            # Tenta aplicar regras ao prefixo PEFT ou nome inferido
            target_names_for_rules = [original_layer_name, peft_prefix.removesuffix(".")]
            for pattern in self._sorted_rule_patterns:
                for target_name in target_names_for_rules:
                    if fnmatch.fnmatch(target_name, pattern):
                        rule_config = self.layer_rules[pattern]
                        rank_to_use = rule_config.get("rank", self.default_rank)
                        alpha_to_use = rule_config.get("alpha", self.default_alpha)
                        rule_applied = f"Rule ('{pattern}')"
                        matched_rule = True
                        break
                if matched_rule:
                    break

            # Args para o construtor do Dummy, usando rank/alpha determinados
            dummy_args = [peft_prefix, None, rank_to_use, alpha_to_use]
            dummy_kwargs = self.global_additional_kwargs.copy()

            try:
                print(
                    f"[LoRA Load] Creating dummy for prefix: {peft_prefix} "
                    f"(Inferred orig: {original_layer_name}) "
                    f"({rule_applied}: Rank={rank_to_use}, Alpha={alpha_to_use})"  # Log atualizado
                )
                dummy_module = self.dummy_klass(*dummy_args, **dummy_kwargs)

                # Passa o remaining_state_dict para que PeftBase.load_state_dict possa remover chaves
                result = dummy_module.load_state_dict(remaining_state_dict, strict=True)

                self.lora_modules[original_layer_name] = dummy_module
                processed_dummy_prefixes.add(peft_prefix)
                loaded_module_prefixes.add(peft_prefix)

                # Chaves do dummy foram removidas de remaining_state_dict
                # Atualiza lista de inesperadas para este prefixo
                keys_still_unexpected = [k for k in remaining_state_dict if k.startswith(peft_prefix)]
                unexpected_keys_overall = [k for k in unexpected_keys_overall if not k.startswith(peft_prefix)]
                unexpected_keys_overall.extend(keys_still_unexpected)

                if result.missing_keys:
                    print(f"Warning: Dummy module {peft_prefix} reported missing keys: {result.missing_keys}")
                    missing_keys_overall.extend([peft_prefix + k for k in result.missing_keys])
                if result.unexpected_keys:
                    print(f"Warning: Dummy module {peft_prefix} reported unexpected keys: {result.unexpected_keys}")
                    unexpected_keys_overall.extend(
                        [peft_prefix + k for k in result.unexpected_keys if (peft_prefix + k) not in unexpected_keys_overall]
                    )

            except Exception as e:
                print(f"Error creating or loading dummy for prefix {peft_prefix}: {e}")

        final_unexpected = [k for k in unexpected_keys_overall if k.startswith(self.prefix)]  # Re-filter just in case
        final_missing = [k for k in missing_keys_overall if k.startswith(self.prefix)]

        if final_unexpected:
            print(f"[LoRA WARNING] Unexpected keys found for prefix '{self.prefix}' after loading:")
            for k in sorted(list(set(final_unexpected))):
                print(f"  - {k}")
        if final_missing:
            print(f"[LoRA WARNING] Missing keys for prefix '{self.prefix}' during state_dict load:")
            for k in sorted(list(set(final_missing))):
                print(f"  - {k}")

    def state_dict(self) -> dict:
        """Returns the state dict containing keys from all managed modules (real and dummy)."""
        state_dict = {}
        for module in self.lora_modules.values():
            module_sd = module.state_dict()
            state_dict.update(module_sd)
        return state_dict

    def parameters(self) -> list[Parameter]:
        """Returns a list of trainable parameters from all *real* PEFT modules."""
        parameters = []
        for module in self.lora_modules.values():
            if not isinstance(module, self.dummy_klass) and module._initialized:
                parameters.extend(list(module.parameters()))
        return parameters

    def requires_grad_(self, requires_grad: bool):
        """Sets requires_grad for all parameters of *real* PEFT modules."""
        count = 0
        for module in self.lora_modules.values():
            if not isinstance(module, self.dummy_klass) and module._initialized:
                if list(module.parameters()):
                    module.requires_grad_(requires_grad)
                    for p in module.parameters():
                        if p.requires_grad == requires_grad:  # Check if state actually changed
                            count += 1

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "LoRAModuleWrapper":
        """Moves all managed modules (real and dummy) to the specified device/dtype."""
        for name, module in self.lora_modules.items():
            try:
                module.to(device=device, dtype=dtype)
                if isinstance(module, DoRAModule) and device is not None and "cuda" in str(device):
                    module.train_device = device
            except Exception as e:
                print(f"Error moving module {name} (prefix {module.prefix}) to {device}/{dtype}: {e}")
        return self

    def modules(self) -> list[nn.Module]:
        """Returns a list of all managed PEFT modules (real and dummy)."""
        return list(self.lora_modules.values())

    def hook_to_module(self):
        """Applies the forward hook to the original modules corresponding to real PEFT modules."""
        if self.orig_module is None:
            print("Warning: Cannot apply hooks without the original root module.")
            return
        hook_count = 0
        for name, module in self.lora_modules.items():
            if (
                not isinstance(module, self.dummy_klass)
                and module._orig_module is not None
                and module._initialized  # Only hook initialized modules
            ):
                try:
                    module.hook_to_module()
                    hook_count += 1
                except Exception as e:
                    print(f"Error applying hook for PEFT module {name} (prefix {module.prefix}): {e}")

    def remove_hook_from_module(self):
        """Removes the forward hook from the original modules."""
        if self.orig_module is None:
            return
        remove_count = 0
        for name, module in self.lora_modules.items():
            if (
                not isinstance(module, self.dummy_klass)
                and module._orig_module is not None
                and module.is_applied  # Only remove if hook is applied
            ):
                try:
                    module.remove_hook_from_module()
                    remove_count += 1
                except Exception as e:
                    print(f"Error removing hook for PEFT module {name} (prefix {module.prefix}): {e}")

    def apply_to_module(self):
        """Applies (merges) PEFT weights directly into the original modules' weights (real modules only)."""
        if self.orig_module is None:
            print("Warning: Cannot apply weights without the original root module.")
            return
        apply_count = 0
        for name, module in self.lora_modules.items():
            if (
                not isinstance(module, self.dummy_klass)
                and module._orig_module is not None
                and module._initialized  # Only apply initialized modules
            ):
                try:
                    module.apply_to_module()  # Implementation is in PeftBase subclasses (TODO)
                    apply_count += 1
                except NotImplementedError:
                    print(f"Warning: apply_to_module not implemented for {type(module).__name__} ({name}).")
                except Exception as e:
                    print(f"Error applying weights for PEFT module {name} (prefix {module.prefix}): {e}")

    def extract_from_module(self, base_module: nn.Module):
        """
        Extracts PEFT weights by comparing the current original module with a base module.
        (Requires implementation in LoRAModule, DoRAModule, etc.)
        """
        if self.orig_module is None:
            print("Warning: Cannot extract weights without the current original root module.")
            return
        extract_count = 0
        for name, module in self.lora_modules.items():
            if not isinstance(module, self.dummy_klass) and module._orig_module is not None:
                try:
                    corresponding_base_submodule = base_module.get_submodule(name)
                    module.extract_from_module(corresponding_base_submodule)  # Implementation in PeftBase subclasses (TODO)
                    extract_count += 1
                except AttributeError:
                    print(f"Warning: Could not find base submodule '{name}' during extraction.")
                except NotImplementedError:
                    print(f"Warning: extract_from_module not implemented for {type(module).__name__} ({name}).")
                except Exception as e:
                    print(f"Error extracting weights for PEFT module {name} (prefix {module.prefix}): {e}")

    def prune(self):
        """Removes all dummy modules from management."""
        initial_count = len(self.lora_modules)
        self.lora_modules = {k: v for (k, v) in self.lora_modules.items() if not isinstance(v, self.dummy_klass)}
        pruned_count = initial_count - len(self.lora_modules)

    def set_dropout(self, dropout_probability: float):
        """Sets the dropout probability for all *real* PEFT modules."""
        if not 0 <= dropout_probability <= 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        count = 0
        for module in self.lora_modules.values():
            if not isinstance(module, self.dummy_klass):
                # Check if the module has a dropout attribute and it's an nn.Dropout instance
                if hasattr(module, "dropout") and isinstance(getattr(module, "dropout", None), nn.Dropout):
                    module.dropout.p = dropout_probability
                    count += 1

    @staticmethod
    def get_block_id_from_key_prefix(lora_module_prefix_with_dot: str) -> str:
        """
        Determines the Block ID (IN00-OUT11, M00, BASE) from the full PEFT module prefix.
        Used for organizing keys. Returns 'BASE' if not specifically mapped.
        """
        lora_module_prefix = lora_module_prefix_with_dot.removesuffix(".")
        sorted_map_prefixes = sorted(KEY_TO_BLOCK_MAPPING.keys(), key=len, reverse=True)
        for map_prefix in sorted_map_prefixes:
            if lora_module_prefix.startswith(map_prefix):
                return KEY_TO_BLOCK_MAPPING[map_prefix]
        return "BASE"

    def generate_keys_by_block_file(self):
        """Generates a text file with all managed PEFT keys organized by UNet block."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"unetKeysByBlock_{timestamp}.txt"
        
        print(f"[LoRA DEBUG] Vai gerar key file com {len(self.lora_modules)} módulos:")
        
        if not self.lora_modules:
            print("Warning: No LoRA/DoRA/LoHa modules found. Key file not generated.")
            return

        print(f"Generating key file by block: {output_filename}")
        from collections import defaultdict

        blocks_data: dict[str, list[str]] = defaultdict(list)
        all_found_block_ids = set()

        for module_instance in self.lora_modules.values():
            module_prefix_with_dot = module_instance.prefix
            block_id = self.get_block_id_from_key_prefix(module_prefix_with_dot)
            all_found_block_ids.add(block_id)

            try:
                module_state_dict = module_instance.state_dict()
                for key in module_state_dict.keys():
                    blocks_data[block_id].append(key)
            except Exception as e:
                print(f"Error getting state_dict for module with prefix {module_prefix_with_dot}: {e}")

        sorted_block_ids = BLOCK_IDS[:]  # respeita exatamente 1 input, 4 down, 1 mid, 4 up, 1 output

        output_lines = []
        for block_id in sorted_block_ids:
            output_lines.append(f"=== {block_id} ===")
            keys_in_block = sorted(blocks_data.get(block_id, []))
            if keys_in_block:
                output_lines.extend(keys_in_block)
            else:
                output_lines.append("(Empty)")
            output_lines.append("")

        try:
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines).strip())
            abs_path = os.path.abspath(output_filename)
            print(f"Successfully generated key file '{output_filename}' at '{abs_path}'")
        except Exception as e:
            print(f"Error writing key file '{output_filename}': {e}")
            traceback.print_exc()
