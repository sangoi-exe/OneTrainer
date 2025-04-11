# -*- coding: utf-8 -*-
# modules/lora/LoRAModule.py

# INÍCIO ALTERAÇÃO: Adicionar imports necessários da versão modificada
import fnmatch
import os
import re

# FIM ALTERAÇÃO
import copy
import math
from abc import abstractmethod
from collections.abc import Mapping

# INÍCIO ALTERAÇÃO: Adicionar imports necessários da versão modificada
from typing import Any

# FIM ALTERAÇÃO

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import PeftType
from modules.util.quantization_util import get_unquantized_weight, get_weight_shape

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# INÍCIO ALTERAÇÃO: Corrigir import do Parameter
from torch.nn import Conv2d, Dropout, Linear, Parameter

# FIM ALTERAÇÃO


# INÍCIO ALTERAÇÃO: Adicionar RuleSet da versão modificada
class RuleSet:
    """Helper class to manage configuration rules based on layer name patterns."""

    def __init__(self, pattern_dict: dict[str, dict[str, int | float]] | None):
        self.patterns = pattern_dict or {}
        # Sort patterns by length descending to match most specific first
        self._sorted_patterns = sorted(self.patterns.keys(), key=len, reverse=True)

    def match(self, name: str) -> dict[str, int | float] | None:
        """Finds the first pattern (most specific to least specific) that matches the name and returns its configuration."""
        for pattern in self._sorted_patterns:
            if fnmatch.fnmatch(name, pattern):
                # Return the configuration dictionary associated with the found pattern
                return self.patterns[pattern]
        return None  # Return None if no pattern matches


# FIM ALTERAÇÃO


# INÍCIO ALTERAÇÃO: Adicionar definições de blocos e mapeamento da versão modificada
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

# Maps common SD UNet layer name prefixes to block IDs
KEY_TO_BLOCK_MAPPING = {
    # Input Blocks
    "lora_unet_conv_in": "IN00",
    "lora_unet_time_embedding": "IN00",  # Often grouped with input conv
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
    "lora_unet_down_blocks_2_downsamplers_0": "IN09",  # Often empty after this
    "lora_unet_down_blocks_3_resnets_0": "IN10",  # XL specific?
    "lora_unet_down_blocks_3_resnets_1": "IN11",  # XL specific?
    # Middle Block
    "lora_unet_mid_block": "M00",  # Covers resnets and attentions
    # Output Blocks
    "lora_unet_up_blocks_0_resnets_0": "OUT03",  # Starts from OUT03 in many architectures
    "lora_unet_up_blocks_0_attentions_0": "OUT03",
    "lora_unet_up_blocks_0_resnets_1": "OUT04",
    "lora_unet_up_blocks_0_attentions_1": "OUT04",
    "lora_unet_up_blocks_0_resnets_2": "OUT05",
    "lora_unet_up_blocks_0_attentions_2": "OUT05",
    "lora_unet_up_blocks_0_upsamplers_0": "OUT05",
    "lora_unet_up_blocks_1_resnets_0": "OUT06",
    "lora_unet_up_blocks_1_attentions_0": "OUT06",
    "lora_unet_up_blocks_1_resnets_1": "OUT07",
    "lora_unet_up_blocks_1_attentions_1": "OUT07",
    "lora_unet_up_blocks_1_resnets_2": "OUT08",
    "lora_unet_up_blocks_1_attentions_2": "OUT08",
    "lora_unet_up_blocks_1_upsamplers_0": "OUT08",
    "lora_unet_up_blocks_2_resnets_0": "OUT09",
    "lora_unet_up_blocks_2_resnets_1": "OUT10",
    "lora_unet_up_blocks_2_resnets_2": "OUT11",
    "lora_unet_up_blocks_2_upsamplers_0": "OUT11",  # Often grouped with last resnet
    # Final Conv
    "lora_unet_conv_out": "OUT11",  # Usually considered part of the last block
    # Text Encoder (Example prefix, adjust if needed)
    "lora_te1_": "BASE",  # Text Encoder 1
    "lora_te2_": "BASE",  # Text Encoder 2 (XL)
}
# FIM ALTERAÇÃO


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
        # INÍCIO ALTERAÇÃO: Usar a sanitização de prefixo da versão modificada
        # Ensure the prefix is safe for file/key names and ends with a dot
        self.prefix = prefix.replace(".", "_") + "."
        # FIM ALTERAÇÃO
        self._orig_module = [orig_module] if orig_module else None
        self.is_applied = False
        self.layer_kwargs = {}
        self._initialized = False

        if orig_module is not None:
            # INÍCIO ALTERAÇÃO: Reverter para a validação estrita de camadas do original
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
                    # Manter a validação estrita do arquivo original
                    raise NotImplementedError(
                        f"Only Linear and Conv2d are supported layers. Got {type(orig_module)} for prefix {self.prefix}"
                    )
            # FIM ALTERAÇÃO

    def hook_to_module(self):
        # INÍCIO ALTERAÇÃO: Adicionar verificação de _orig_module do modificado
        if self._orig_module is None:
            # Cannot hook if there is no original module (e.g., dummy loaded from state dict)
            # print(f"Warning: Cannot hook {self.prefix}: No original module set.")
            return
        # FIM ALTERAÇÃO
        if not self.is_applied:
            self.orig_forward = self.orig_module.forward
            self.orig_train = self.orig_module.train
            self.orig_eval = self.orig_module.eval
            self.orig_module.forward = self.forward
            self.orig_module.train = self._wrap_train
            self.orig_module.eval = self._wrap_eval
            self.is_applied = True

    def remove_hook_from_module(self):
        # INÍCIO ALTERAÇÃO: Adicionar verificação de _orig_module e orig_forward
        if self._orig_module is None or self.orig_forward is None:
            # print(f"Warning: Cannot remove hook from {self.prefix}: Module or original forward not available.")
            return
        # FIM ALTERAÇÃO
        assert self.orig_forward is not None  # Mantido do original
        if self.is_applied:
            self.orig_module.forward = self.orig_forward
            self.orig_module.train = self.orig_train
            self.orig_module.eval = self.orig_eval
            self.is_applied = False

    def _wrap_train(self, mode=True):
        # INÍCIO ALTERAÇÃO: Adicionar verificação de _orig_module e orig_train
        if self._orig_module is None or self.orig_train is None:
            return
        # FIM ALTERAÇÃO
        self.orig_train(mode)
        self.train(mode)

    def _wrap_eval(self):
        # INÍCIO ALTERAÇÃO: Adicionar verificação de _orig_module e orig_eval
        if self._orig_module is None or self.orig_eval is None:
            return
        # FIM ALTERAÇÃO
        self.orig_eval()
        self.eval()

    def make_weight(self, A: Tensor, B: Tensor):
        """Layer-type-independent way of creating a weight matrix from LoRA A/B."""
        # INÍCIO ALTERAÇÃO: Adicionar verificação de shape
        if self.shape is None:
            raise RuntimeError(
                f"Cannot make weight for {self.prefix}: shape not determined (likely unsupported layer type)."
            )
        # FIM ALTERAÇÃO
        W = B.view(B.size(0), -1) @ A.view(A.size(0), -1)
        return W.view(self.shape)

    def check_initialized(self):
        """Checks, and raises an exception, if the module is not initialized."""
        if not self._initialized:
            raise RuntimeError(f"Module {self.prefix} is not initialized.")

        # Perform assertions to make pytype happy.
        # INÍCIO ALTERAÇÃO: Tornar asserts mais informativos
        assert self._orig_module is not None, f"orig_module is None for {self.prefix}"
        assert self.orig_forward is not None, f"orig_forward is None for {self.prefix}"
        # Original tinha assert self.op is not None, mas agora op pode ser None temporariamente
        # A verificação será feita no forward específico de cada subclasse
        # FIM ALTERAÇÃO

    @property
    def orig_module(self) -> nn.Module:
        # INÍCIO ALTERAÇÃO: Usar a verificação aprimorada
        if self._orig_module is None:
            raise AttributeError(
                f"Original module not set for PEFT layer with prefix {self.prefix}"
            )
        # FIM ALTERAÇÃO
        return self._orig_module[0]

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # INÍCIO ALTERAÇÃO: Usar a lógica de filtragem de prefixo da versão modificada,
        # mas inicializar pesos *antes* de chamar super().load_state_dict
        # Filter the state_dict to contain only keys relevant to this specific module
        module_prefix_with_dot = self.prefix  # self.prefix already ends with '.'
        relevant_keys = {
            k: v for k, v in state_dict.items() if k.startswith(module_prefix_with_dot)
        }

        if not relevant_keys:
            # If no relevant keys are found, this module might not be in the state_dict,
            # or the prefix is incorrect. Return empty keys to indicate this.
            # Do not initialize the module if there are no keys to load.
            # print(f"Debug: No keys found for prefix '{module_prefix_with_dot}' in state_dict for {type(self).__name__}.")
            # Return what super() would return in this case (handled by LoRAModuleWrapper)
            return nn.modules.module._IncompatibleKeys(
                [], []
            )  # No missing, no unexpected

        # Remove the module prefix from the keys before calling super().load_state_dict
        state_dict_local = {
            k.removeprefix(module_prefix_with_dot): v for k, v in relevant_keys.items()
        }

        # Initialize weights only if they haven't been initialized, there are keys,
        # and it's a supported layer type (has _orig_module set during __init__)
        if not self._initialized and state_dict_local and self._orig_module is not None:
            try:
                # Ensure rank/alpha etc. are available if needed by initialize_weights
                self.initialize_weights()  # Call the subclass implementation
                self._initialized = True  # Mark as initialized *after* success
            except NotImplementedError as e:
                print(
                    f"Error initializing weights for {self.prefix}: {e}. Module might not load correctly."
                )
                # Do not mark as initialized if initialization fails
            except AttributeError as e:
                print(
                    f"Error during weight initialization for {self.prefix} (likely missing rank/alpha): {e}. Module might not load correctly."
                )

        # Attempt to load the local state_dict only if initialized
        load_result = nn.modules.module._IncompatibleKeys(
            [], []
        )  # Default empty result
        if self._initialized:
            try:
                load_result = super().load_state_dict(
                    state_dict_local, strict=strict, assign=assign
                )
            except Exception as e:
                print(f"Error calling super().load_state_dict for {self.prefix}: {e}")
                # Keep default load_result indicating potential issues
        else:
            # If not initialized (or failed), return indicating nothing was loaded for this module
            missing_keys = list(state_dict_local.keys())
            load_result = nn.modules.module._IncompatibleKeys(missing_keys, [])

        # Crucial: Remove the processed keys from the *original* state_dict passed to the wrapper
        # This prevents dummy modules from trying to load them again.
        keys_processed_or_skipped = list(relevant_keys.keys())
        for key in keys_processed_or_skipped:
            if key in state_dict:  # Check if the key still exists (it should)
                state_dict.pop(key)  # Remove keys that were handled (or attempted) here

        return load_result
        # FIM ALTERAÇÃO

    # INÍCIO ALTERAÇÃO: Adotar o state_dict modificado que adiciona o prefixo
    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        # Get the local state_dict (without the module's own prefix)
        local_state_dict = super().state_dict(
            *args, destination=None, prefix="", keep_vars=keep_vars
        )

        # Add the module's prefix (self.prefix) to each local key
        state_dict_with_prefix = {
            self.prefix + k: v for k, v in local_state_dict.items()
        }

        # Handle the standard 'prefix' argument from PyTorch's state_dict method
        # (this prefix usually comes from nested calls, like model.state_dict())
        if prefix:
            state_dict_with_prefix = {
                prefix + k: v for k, v in state_dict_with_prefix.items()
            }

        # Update the destination if provided, otherwise return the dictionary with prefixes
        if destination is None:
            destination = state_dict_with_prefix
        else:
            destination.update(state_dict_with_prefix)

        return destination

    # FIM ALTERAÇÃO

    @abstractmethod
    def initialize_weights(self):
        # Revertido: A inicialização será feita nas subclasses como no original
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
        # INÍCIO ALTERAÇÃO: Usar a verificação de orig_module e rank aprimorada
        if self._orig_module is None:
            raise RuntimeError(
                f"Cannot create layer for {self.prefix}: Original module is None."
            )
        if not hasattr(self, "rank"):
            raise AttributeError(
                f"Rank not set for PEFT module {self.prefix} before calling create_layer."
            )
        # FIM ALTERAÇÃO

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
                # INÍCIO ALTERAÇÃO: Manter a nota sobre grupos e stride/padding explícitos de Conv2d
                # Note: small departure here from part of the community.
                # The original Microsoft repo does it this way. The cloneofsimo
                # repo handles the groups in lora_down. We follow the Microsoft
                # way. In reality, there shouldn't be any difference.
                lora_up = Conv2d(
                    self.rank,
                    out_channels // groups,
                    (1, 1),
                    stride=1,
                    padding=0,
                    bias=False,
                    device=device,
                )
                # FIM ALTERAÇÃO

            case _:
                # This case should ideally not be reached due to the check in __init__
                raise NotImplementedError(
                    f"Layer creation not implemented for type: {type(self.orig_module)}"
                )

        return lora_down, lora_up

    @classmethod
    def make_dummy(cls):
        """Create a dummy version of a PEFT class."""

        # INÍCIO ALTERAÇÃO: Usar a implementação Dummy da versão modificada, que parece mais robusta
        # para state_dict e inicialização.
        class Dummy(cls):
            def __init__(self, *args, **kwargs):
                # Extract prefix carefully from args or kwargs
                prefix_arg = args[0] if args else kwargs.get("prefix", "dummy_prefix")
                # Call PeftBase init with None for orig_module
                PeftBase.__init__(self, prefix_arg, None)
                self._state_dict = {}
                # Dummies are not 'initialized' in the sense of having real weights,
                # but we set _initialized based on whether state_dict was loaded.
                self._initialized = False
                # Store rank/alpha if provided, needed for state_dict compatibility sometimes
                self.rank = args[2] if len(args) > 2 else kwargs.get("rank", 0)
                self.alpha_val = (
                    args[3] if len(args) > 3 else kwargs.get("alpha", 0.0)
                )  # Store alpha value
                # Dummies don't have dropout in the same way, but add attribute for consistency
                self.dropout = Dropout(0)  # Dummy dropout

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
                # Logic from modified version to handle prefixes correctly
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
                    # No keys for this dummy, return empty incompatibility info
                    return nn.modules.module._IncompatibleKeys([], [])

                # Store the loaded state locally, mark as "initialized" to have a state_dict
                self._initialized = True
                self._state_dict = copy.deepcopy(state_dict_local)
                # Register alpha buffer if it was loaded
                if "alpha" in self._state_dict:
                    # Ensure buffer exists before loading into it
                    if not hasattr(self, "alpha"):
                        self.register_buffer("alpha", self._state_dict["alpha"].clone())
                    else:
                        self.alpha.copy_(self._state_dict["alpha"])
                    # Dummies don't train, so no grad needed
                    self.alpha.requires_grad_(False)
                elif self.alpha_val is not None:  # If alpha value was passed at init
                    if not hasattr(self, "alpha"):
                        self.register_buffer("alpha", torch.tensor(self.alpha_val))
                        self.alpha.requires_grad_(False)

                # Remove processed keys from the input dict
                keys_to_remove = list(relevant_keys.keys())
                for key in keys_to_remove:
                    if key in state_dict:
                        state_dict.pop(key)

                # Report no missing/unexpected keys for the dummy itself
                return nn.modules.module._IncompatibleKeys([], [])

            # Use the state_dict from the modified PeftBase which handles prefixes
            # Inherited method PeftBase.state_dict(...) works correctly here now.

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
        # FIM ALTERAÇÃO


class LoHaModule(PeftBase):
    """Implementation of LoHa from Lycoris."""

    rank: int
    dropout: Dropout
    # INÍCIO ALTERAÇÃO: Usar Parameter para pesos LoHa
    hada_w1_a: Parameter | None
    hada_w1_b: Parameter | None
    hada_w2_a: Parameter | None
    hada_w2_b: Parameter | None
    # FIM ALTERAÇÃO

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
            # INÍCIO ALTERAÇÃO: Inicializar pesos aqui como no original
            try:
                self.initialize_weights()
                self.alpha = self.alpha.to(orig_module.weight.device)
            except NotImplementedError:
                # Should not happen if __init__ checks passed, but good practice
                print(
                    f"Warning: LoHa init failed for {prefix} due to unsupported layer type."
                )
            # FIM ALTERAÇÃO
        # INÍCIO ALTERAÇÃO: Mover requires_grad_(False) para após a inicialização
        if hasattr(self, "alpha"):
            self.alpha.requires_grad_(False)
        # FIM ALTERAÇÃO

    def initialize_weights(self):
        # INÍCIO ALTERAÇÃO: Lógica de inicialização movida para cá (como no original)
        if self._initialized:
            return
        if self._orig_module is None:
            return  # Cannot initialize without original module

        hada_w1_b_module, hada_w1_a_module = self.create_layer()
        hada_w2_b_module, hada_w2_a_module = self.create_layer()

        # Assign as nn.Parameter
        self.hada_w1_a = Parameter(hada_w1_a_module.weight)
        self.hada_w1_b = Parameter(hada_w1_b_module.weight)
        self.hada_w2_a = Parameter(hada_w2_a_module.weight)
        self.hada_w2_b = Parameter(hada_w2_b_module.weight)

        # Initialization values from original
        nn.init.normal_(self.hada_w1_a, std=0.1)
        nn.init.normal_(self.hada_w1_b, std=1)
        nn.init.constant_(self.hada_w2_a, 0)
        nn.init.normal_(self.hada_w2_b, std=1)
        self._initialized = True
        # FIM ALTERAÇÃO

    def check_initialized(self):
        super().check_initialized()
        assert self.hada_w1_a is not None
        assert self.hada_w1_b is not None
        assert self.hada_w2_a is not None
        assert self.hada_w2_b is not None
        # INÍCIO ALTERAÇÃO: Adicionar verificação de alpha
        assert hasattr(self, "alpha"), f"Alpha buffer missing in LoHa {self.prefix}"
        # FIM ALTERAÇÃO

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        # INÍCIO ALTERAÇÃO: Adicionar verificação de self.op
        if self.op is None:  # If the original layer was not supported
            print(
                f"Warning: Skipping LoHa forward for {self.prefix} (unsupported layer type). Returning original output."
            )
            return self.orig_forward(x)
        # FIM ALTERAÇÃO

        # Logic from original
        W1 = self.make_weight(
            self.dropout(self.hada_w1_b), self.dropout(self.hada_w1_a)
        )
        W2 = self.make_weight(
            self.dropout(self.hada_w2_b), self.dropout(self.hada_w2_a)
        )
        # INÍCIO ALTERAÇÃO: Usar .item() para alpha no cálculo
        scale = self.alpha.item() / self.rank
        W = (W1 * W2) * scale
        # FIM ALTERAÇÃO
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
            # INÍCIO ALTERAÇÃO: Inicializar pesos aqui como no original
            try:
                self.initialize_weights()
                self.alpha = self.alpha.to(orig_module.weight.device)
            except NotImplementedError:
                print(
                    f"Warning: LoRA init failed for {prefix} due to unsupported layer type."
                )
            # FIM ALTERAÇÃO
        # INÍCIO ALTERAÇÃO: Mover requires_grad_(False) para após a inicialização
        if hasattr(self, "alpha"):
            self.alpha.requires_grad_(False)
        # FIM ALTERAÇÃO

    def initialize_weights(self):
        # INÍCIO ALTERAÇÃO: Lógica de inicialização movida para cá (como no original)
        if self._initialized:
            return
        if self._orig_module is None:
            return

        self.lora_down, self.lora_up = self.create_layer()
        # Initialization from original
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self._initialized = True
        # FIM ALTERAÇÃO

    def check_initialized(self):
        super().check_initialized()
        assert (
            self.lora_down is not None
        ), f"LoRA down not initialized for {self.prefix}"
        assert self.lora_up is not None, f"LoRA up not initialized for {self.prefix}"
        # INÍCIO ALTERAÇÃO: Adicionar verificação de alpha
        assert hasattr(self, "alpha"), f"Alpha buffer missing in LoRA {self.prefix}"
        # FIM ALTERAÇÃO

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        # INÍCIO ALTERAÇÃO: Adicionar verificação de self.op e usar .item() para alpha
        if self.op is None:  # If the original layer was not supported
            print(
                f"Warning: Skipping LoRA forward for {self.prefix} (unsupported layer type). Returning original output."
            )
            return self.orig_forward(x)

        lora_output = self.lora_up(self.lora_down(self.dropout(x)))
        original_output = self.orig_forward(x)
        scale = self.alpha.item() / self.rank
        return original_output + lora_output * scale
        # FIM ALTERAÇÃO

    def apply_to_module(self):
        # TODO: Implement merging logic if needed
        raise NotImplementedError

    def extract_from_module(self, base_module: nn.Module):
        # TODO: Implement extraction logic if needed
        raise NotImplementedError


class DoRAModule(LoRAModule):
    """Weight-decomposed low rank adaptation."""

    # INÍCIO ALTERAÇÃO: Manter atributos da versão modificada
    # dora_num_dims is implicitly handled by norm calculation now
    dora_scale: Parameter | None  # Use Parameter for trainable scale
    norm_epsilon: bool
    train_device: torch.device  # Add train_device attribute
    # FIM ALTERAÇÃO

    # INÍCIO ALTERAÇÃO: Modificar init para aceitar kwargs de DoRA
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

    # FIM ALTERAÇÃO

    def initialize_weights(self):
        # INÍCIO ALTERAÇÃO: Chamar init de LoRA e depois adicionar init de DoRA
        if self._initialized:
            return
        if self._orig_module is None:
            return

        # 1. Initialize LoRA components first
        super().initialize_weights()  # Calls create_layer, inits lora_down/up
        if not self._initialized:
            return  # Stop if LoRA init failed

        # 2. Initialize DoRA specific scale
        # Use float32 for norm calculation stability, then cast back
        orig_weight = get_unquantized_weight(
            self.orig_module, torch.float32, self.train_device
        )
        eps = torch.finfo(orig_weight.dtype).eps if self.norm_epsilon else 0.0

        # Calculate norm based on layer type (more explicit version from modified file)
        if isinstance(self.orig_module, nn.Conv2d):
            # Norm per output filter (output channel dim 0)
            norm = (
                torch.linalg.vector_norm(
                    orig_weight, ord=2, dim=(1, 2, 3), keepdim=True
                )
                + eps
            )
        elif isinstance(self.orig_module, nn.Linear):
            # Norm per output neuron (output channel dim 0) -> weight dims are (out, in)
            norm = (
                torch.linalg.vector_norm(orig_weight, ord=2, dim=1, keepdim=True) + eps
            )
            # Alternative for Linear (like original): norm per column (input feature)
            # norm = torch.linalg.vector_norm(orig_weight, ord=2, dim=0, keepdim=True) + eps
            # Let's stick to norm per output neuron, seems more common for magnitude scaling
        else:
            # Fallback or should not happen due to initial checks
            print(
                f"Warning: Using generic L2 norm for unsupported layer type {type(self.orig_module)} in DoRA init for {self.prefix}"
            )
            norm = (
                torch.linalg.vector_norm(orig_weight.flatten(), ord=2) + eps
            )  # Norm of flattened tensor
            # Reshape might be needed depending on how scale is used, but this is fallback
            norm = norm.reshape(1 for _ in orig_weight.dim())  # Make it broadcastable

        # Create scale Parameter on the correct device and dtype
        self.dora_scale = Parameter(
            norm.to(
                device=self.orig_module.weight.device,
                dtype=self.orig_module.weight.dtype,
            )
        )
        # self._initialized is already True from super().initialize_weights()
        del orig_weight
        # FIM ALTERAÇÃO

    def check_initialized(self):
        super().check_initialized()
        assert (
            self.dora_scale is not None
        ), f"DoRA scale not initialized for {self.prefix}"

    def forward(self, x, *args, **kwargs):
        self.check_initialized()
        # INÍCIO ALTERAÇÃO: Usar a lógica forward aprimorada da versão modificada
        # que aplica dropout no input e calcula norma corretamente
        if self.op is None:  # If the original layer was not supported
            print(
                f"Warning: Skipping DoRA forward for {self.prefix} (unsupported layer type). Returning original output."
            )
            # Still need to apply dropout if it's part of the DoRA spec
            return self.orig_forward(
                self.dropout(x)
            )  # Apply dropout as per DoRA paper mention

        # Get components
        A = self.lora_down.weight
        B = self.lora_up.weight
        # Calculate LoRA delta, use float32 for intermediate calcs if needed
        lora_dtype = A.dtype
        orig_weight = get_unquantized_weight(
            self.orig_module, lora_dtype, self.train_device
        )

        # Calculate the LoRA modification scaled by alpha/rank
        lora_delta_w = self.make_weight(A, B) * (self.alpha.item() / self.rank)

        # Calculate the adapted weight matrix W' = W + deltaW
        adapted_weight = orig_weight + lora_delta_w

        # Calculate the norm of the adapted weight matrix W'
        # Use float32 for stability, detach for backprop as per paper
        adapted_weight_f32 = adapted_weight.to(torch.float32)
        eps = torch.finfo(adapted_weight_f32.dtype).eps if self.norm_epsilon else 0.0

        if isinstance(self.orig_module, nn.Conv2d):
            norm = (
                torch.linalg.vector_norm(
                    adapted_weight_f32.detach(), ord=2, dim=(1, 2, 3), keepdim=True
                )
                + eps
            )
        elif isinstance(self.orig_module, nn.Linear):
            norm = (
                torch.linalg.vector_norm(
                    adapted_weight_f32.detach(), ord=2, dim=1, keepdim=True
                )
                + eps
            )  # Per output neuron
            # norm = torch.linalg.vector_norm(adapted_weight_f32.detach(), ord=2, dim=0, keepdim=True) + eps # Per input feature (alternative)
        else:
            # Fallback - should not happen
            norm = (
                torch.linalg.vector_norm(adapted_weight_f32.detach().flatten(), ord=2)
                + eps
            )
            norm = norm.reshape(1 for _ in adapted_weight_f32.dim())

        # Normalize the adapted weight and scale by DoRA magnitude
        # Ensure norm is on the correct device and dtype before division
        norm = norm.to(device=adapted_weight.device, dtype=adapted_weight.dtype)
        final_weight = self.dora_scale * (adapted_weight / norm)

        # Apply dropout to input x as mentioned in DoRA implementations
        x_dropout = self.dropout(x)

        # Perform the original operation with the final DoRA weight
        output = self.op(
            x_dropout,
            final_weight,
            self.orig_module.bias,  # Use original bias
            **self.layer_kwargs,
        )

        del (
            orig_weight,
            lora_delta_w,
            adapted_weight,
            adapted_weight_f32,
            norm,
            final_weight,
        )  # Memory cleanup
        return output
        # FIM ALTERAÇÃO


# INÍCIO ALTERAÇÃO: (Re)definir Dummies após classes base
DummyLoRAModule = LoRAModule.make_dummy()
DummyDoRAModule = DoRAModule.make_dummy()
DummyLoHaModule = LoHaModule.make_dummy()
# FIM ALTERAÇÃO


# INÍCIO ALTERAÇÃO: Adotar LoRAModuleWrapper da versão modificada, que inclui
# presets, filtros, RuleSet, e lógica de criação/carregamento aprimorada.
class LoRAModuleWrapper:
    """
    Manages PEFT modules (LoRA, LoHa, DoRA) for a PyTorch module.

    Allows applying different rank/alpha configurations to different layers
    based on name patterns defined in `config.lora_layer_patterns` or presets.
    Also supports `module_filter` and `lora_layers_blacklist` for layer selection.
    """

    orig_module: nn.Module | None  # Can be None if only loading from state_dict
    prefix: str
    peft_type: PeftType
    config: TrainConfig  # Store config for access to various settings
    default_rank: int
    default_alpha: float
    lora_layer_patterns: dict[
        str, dict[str, int | float]
    ]  # Final patterns after preset merge
    ruleset: RuleSet  # Applies the patterns
    inclusion_filter_patterns: list[str]  # Patterns for layers to *potentially* include
    exclusion_filter_patterns: list[str]  # Patterns for layers to *definitely* exclude
    klass: type[PeftBase]  # PEFT class to use (LoRA, LoHa, DoRA)
    dummy_klass: type[PeftBase]  # Corresponding Dummy class
    global_additional_kwargs: dict  # Kwargs for PEFT class (e.g., for DoRA)
    lora_modules: dict[
        str, PeftBase
    ]  # Managed PEFT modules (real and dummies), keyed by original layer name

    def __init__(
        self,
        orig_module: nn.Module | None,
        prefix: str,
        config: TrainConfig,
        preset_name: str | None = None,  # Name of the preset to load from PRESETS
        external_module_filter: list[str] | None = None,  # Additional/override filter
        # Removed module_filter arg, now handled by external_module_filter and preset logic
    ):
        """
        Initializes the Wrapper.

        Args:
            orig_module: The original nn.Module to be adapted (can be None).
            prefix: Prefix added to PEFT module keys (e.g., "lora_unet").
            config: TrainConfig object containing settings:
                - peft_type: PeftType (LORA, LOHA, DORA).
                - lora_rank: Default rank if no pattern matches.
                - lora_alpha: Default alpha if no pattern matches.
                - lora_layer_patterns: Base dictionary mapping layer name patterns
                  to {'rank': R, 'alpha': A} overrides.
                - lora_layers_blacklist: List of patterns to explicitly exclude.
                - lora_decompose: True to use DoRA instead of LoRA.
                - lora_decompose_norm_epsilon: Epsilon for DoRA norm.
                - train_device: Device for DoRA calculations.
            preset_name: Optional name of a preset configuration (e.g., 'attn', 'full').
                         Presets define inclusion filters and can override ranks/alphas.
            external_module_filter: Optional list of patterns. If provided, acts as an
                                   *additional* inclusion filter alongside the preset's filter.
        """
        if orig_module is None:
            print(
                f"Info: LoRAModuleWrapper '{prefix}' initialized without an original module. Will only load from state_dict."
            )
        self.orig_module = orig_module
        self.prefix = prefix  # The user-provided prefix (e.g., "lora_unet")
        self.peft_type = config.peft_type
        self.config = config  # Keep a reference to the config

        # Store default values
        self.default_rank = config.lora_rank
        self.default_alpha = config.lora_alpha

        # --- Logic for Presets, Filters, and Patterns (from modified version) ---

        # 1. Load presets and patterns
        # Import PRESETS locally to avoid circular dependencies if PRESETS imports this module
        try:
            # Assume PRESETS is accessible in the project structure
            # Adjust the import path if necessary
            from modules.model_setup.preset_lora import PRESETS
        except ImportError:
            print("Warning: PRESETS dictionary not found. Presets will not be loaded.")
            PRESETS = {}

        # Start with base patterns from main config
        self.lora_layer_patterns = (
            config.lora_layer_patterns.copy() if config.lora_layer_patterns else {}
        )
        preset_inclusion_patterns = []  # Inclusion patterns defined *by the preset*

        if preset_name:
            preset_config = PRESETS.get(preset_name)
            if preset_config is None and preset_name != "full":
                print(
                    f"[LoRA WARNING] Preset '{preset_name}' not found or is None. Using global patterns and external filter/blacklist."
                )
            elif preset_name == "full":
                print(
                    f"[LoRA INFO] Preset 'full' selected. No preset-specific inclusion filter or rank/alpha overrides applied."
                )
                # 'full' implies no *preset* filter, but external filter and blacklist still apply.
            elif isinstance(preset_config, dict):
                print(
                    f"[LoRA INFO] Using preset '{preset_name}' with {len(preset_config)} rules/overrides."
                )
                # Keys of the preset dict act as inclusion patterns *for this preset*
                preset_inclusion_patterns = list(preset_config.keys())
                # Merge preset's rank/alpha overrides into the main patterns
                self.lora_layer_patterns.update(preset_config)
            else:
                print(
                    f"[LoRA WARNING] Preset '{preset_name}' has unexpected format ({type(preset_config)}). Ignoring preset."
                )
        else:
            print(
                "[LoRA INFO] No preset specified. Using global patterns and external filter/blacklist."
            )

        # 2. Combine inclusion filters (Preset + External)
        # A module must match *at least one* inclusion pattern to be considered.
        combined_inclusion_patterns = set(
            p.strip() for p in preset_inclusion_patterns if p.strip()
        )
        if external_module_filter:
            combined_inclusion_patterns.update(
                p.strip() for p in external_module_filter if p.strip()
            )

        # Final list of inclusion patterns (empty means "include everything" that isn't blacklisted)
        self.inclusion_filter_patterns = list(combined_inclusion_patterns)
        print(
            f"[LoRA INFO] Active inclusion patterns: {self.inclusion_filter_patterns if self.inclusion_filter_patterns else 'None (include all unless blacklisted)'}"
        )

        # 3. Prepare exclusion filter (Blacklist)
        self.exclusion_filter_patterns = [
            b.strip() for b in (config.lora_layers_blacklist or []) if b.strip()
        ]
        print(
            f"[LoRA INFO] Active exclusion patterns (blacklist): {self.exclusion_filter_patterns if self.exclusion_filter_patterns else 'None'}"
        )

        # 4. Initialize RuleSet with the *final* combined patterns (global + preset overrides)
        # RuleSet is used ONLY to determine rank/alpha for modules that pass the filters.
        self.ruleset = RuleSet(self.lora_layer_patterns)
        print(
            f"[LoRA INFO] RuleSet initialized with {len(self.ruleset.patterns)} rank/alpha rules."
        )
        # Debug: Show some loaded rules
        # print(f"[LoRA DEBUG] First 5 rules in RuleSet: {list(self.ruleset.patterns.items())[:5]}")

        # 5. Define PEFT classes and global kwargs based on config
        self.global_additional_kwargs = {}
        if self.peft_type == PeftType.LORA:
            if config.lora_decompose:
                self.klass = DoRAModule
                self.dummy_klass = DummyDoRAModule
                self.global_additional_kwargs = {
                    "norm_epsilon": config.lora_decompose_norm_epsilon,
                    "train_device": torch.device(
                        config.train_device
                    ),  # Use config directly
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

        # 6. Create the PEFT modules (uses the filters and ruleset)
        # This is now done in a separate method for clarity
        self.lora_modules = self._initialize_peft_modules(orig_module)
        print(
            f"[LoRA INFO] LoRAModuleWrapper '{self.prefix}' initialized with {len(self.lora_modules)} PEFT modules."
        )

    def _should_include_module(
        self, original_module_name: str, potential_peft_prefix_with_dot: str
    ) -> bool:
        """
        Checks if a module should be included based on its original name and potential PEFT prefix,
        considering exclusion and inclusion filters.
        """
        # 1. Check exclusion filter (blacklist) against both names
        for pattern in self.exclusion_filter_patterns:
            # Check if pattern matches original name OR potential PEFT prefix (without trailing dot for fnmatch flexibility)
            peft_prefix_no_dot = potential_peft_prefix_with_dot.removesuffix(".")
            if fnmatch.fnmatch(original_module_name, pattern) or fnmatch.fnmatch(
                peft_prefix_no_dot, pattern
            ):
                # print(f"[LoRA FILTER DEBUG] Module '{original_module_name}' (PEFT: '{peft_prefix_no_dot}') EXCLUDED by blacklist pattern '{pattern}'") # Optional Debug
                return False  # Explicitly excluded

        # 2. Check inclusion filter against both names (usually matching original name is sufficient, but check both for flexibility)
        if not self.inclusion_filter_patterns:
            # If no inclusion filters are defined, include everything not blacklisted
            # print(f"[LoRA FILTER DEBUG] Module '{original_module_name}' INCLUDED (no inclusion filters & not blacklisted)") # Optional Debug
            return True
        else:
            # If inclusion filters exist, it must match at least one (checking both names)
            peft_prefix_no_dot = potential_peft_prefix_with_dot.removesuffix(".")
            for pattern in self.inclusion_filter_patterns:
                if fnmatch.fnmatch(original_module_name, pattern) or fnmatch.fnmatch(
                    peft_prefix_no_dot, pattern
                ):
                    # print(f"[LoRA FILTER DEBUG] Module '{original_module_name}' (PEFT: '{peft_prefix_no_dot}') INCLUDED by inclusion pattern '{pattern}'") # Optional Debug
                    return True  # Matched an inclusion pattern

        # If inclusion filters exist but none matched either name
        # print(f"[LoRA FILTER DEBUG] Module '{original_module_name}' (PEFT: '{potential_peft_prefix_with_dot}') EXCLUDED (did not match any inclusion pattern)") # Optional Debug
        return False

    def _initialize_peft_modules(
        self, root_module: nn.Module | None
    ) -> dict[str, PeftBase]:
        """
        Identifies, filters, and creates the actual PEFT modules.
        """
        lora_modules: dict[str, PeftBase] = {}
        if root_module is None:
            print(
                "[LoRA WARNING] _initialize_peft_modules called without root_module. No PEFT modules created."
            )
            return lora_modules

        print("[LoRA INFO] Identifying and creating PEFT modules...")
        modules_created_count = 0
        modules_skipped_type = 0
        modules_skipped_filter = 0

        # 1. Iterate through named modules to find candidates
        for name, child_module in root_module.named_modules():
            # Only consider Linear and Conv2d layers
            if not isinstance(child_module, (Linear, Conv2d)):
                modules_skipped_type += 1
                continue  # Skip unsupported layer types

            # Calculate potential PEFT prefix before calling _should_include_module
            # Ensure the name part is sanitized like in PeftBase.__init__ and key generation
            sanitized_name_part = name.replace(".", "_")
            potential_peft_prefix_with_dot = f"{self.prefix}_{sanitized_name_part}."

            # Apply combined filtering logic (inclusion + exclusion)
            # INÍCIO ALTERAÇÃO: Corrigir a chamada para _should_include_module
            # Passar explicitamente ambos os argumentos: name e potential_peft_prefix_with_dot
            if self._should_include_module(name, potential_peft_prefix_with_dot):
                # FIM ALTERAÇÃO
                # Module passed filters, proceed to create PEFT layer

                # Use the already calculated potential_peft_prefix for the actual module instance
                peft_module_prefix = potential_peft_prefix_with_dot.removesuffix(
                    "."
                )  # Remove dot for constructor arg

                # Determine Rank and Alpha using RuleSet
                # Match RuleSet against both names for flexibility
                rank_to_use = self.default_rank
                alpha_to_use = self.default_alpha
                matched_config = self.ruleset.match(name)  # Try original name first
                if matched_config is None:
                    # If original name didn't match, try the potential PEFT prefix (without dot)
                    matched_config = self.ruleset.match(
                        potential_peft_prefix_with_dot.removesuffix(".")
                    )

                rule_source = "(Default)"
                if matched_config:
                    rank_to_use = matched_config.get("rank", self.default_rank)
                    alpha_to_use = matched_config.get("alpha", self.default_alpha)
                    rule_source = f"(Rule: {matched_config})"
                    # print(f"[LoRA RULE DEBUG] Match for '{name}'/'{potential_peft_prefix_with_dot.removesuffix('.')}': {matched_config} -> Rank={rank_to_use}, Alpha={alpha_to_use}") # Optional Debug

                # Prepare args and kwargs for the PEFT class constructor
                args_for_this_module = [
                    peft_module_prefix,
                    child_module,
                    rank_to_use,
                    alpha_to_use,
                ]
                kwargs_for_this_module = self.global_additional_kwargs.copy()

                # Create the PEFT instance
                try:
                    log_msg = (
                        f"[LoRA CREATE] Creating {self.klass.__name__} for: {name} "
                        f"{rule_source} -> Rank={rank_to_use}, Alpha={alpha_to_use} "
                        f"| PEFT Prefix: {potential_peft_prefix_with_dot}"
                    )  # Log with dot
                    # print(log_msg) # Optional detailed log

                    lora_modules[name] = self.klass(
                        *args_for_this_module, **kwargs_for_this_module
                    )
                    modules_created_count += 1

                except Exception as e:
                    print(
                        f"[LoRA ERROR] Failed to create PEFT module for layer '{name}' (prefix {peft_module_prefix}) "
                        f"with Rank={rank_to_use}, Alpha={alpha_to_use}: {e}"
                    )

            else:
                modules_skipped_filter += 1
                # print(f"[LoRA FILTER] Skipping module (failed filter): {name}") # Optional Debug

        print(f"[LoRA INFO] Module Creation Summary for '{self.prefix}':")
        print(f"  - Created: {modules_created_count} PEFT modules.")
        print(f"  - Skipped (Filter): {modules_skipped_filter} modules.")
        print(f"  - Skipped (Type): {modules_skipped_type} non-Linear/Conv2d modules.")
        return lora_modules

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        """
        Loads the state dict into managed PEFT modules (real and dummy).

        Handles creating dummy modules for keys present in the state_dict
        but not corresponding to any initially created real PEFT modules.
        """
        if not self.lora_modules and self.orig_module is None:
            print(
                "[LoRA WARNING] load_state_dict called with no original module and no pre-existing PEFT modules. Attempting to load all as dummies."
            )
            # If there's no orig_module, we can *only* create dummies.

        # Create a copy to modify while iterating
        remaining_state_dict = copy.deepcopy(state_dict)
        loaded_module_prefixes = (
            set()
        )  # Track PEFT prefixes successfully loaded (real or dummy)
        missing_keys_overall = []
        unexpected_keys_overall = list(
            remaining_state_dict.keys()
        )  # Start assuming all are unexpected

        # 1. Try loading into existing *real* modules first
        # Iterate over a copy of keys, as lora_modules might change if dummies are added later
        current_real_module_names = [
            name
            for name, mod in self.lora_modules.items()
            if not isinstance(mod, self.dummy_klass)
        ]
        for name in current_real_module_names:
            module = self.lora_modules.get(name)
            if module is None:
                continue  # Should not happen, but safety check

            try:
                # PeftBase.load_state_dict handles prefix filtering and removal from remaining_state_dict
                result = module.load_state_dict(
                    remaining_state_dict, strict=True
                )  # Use strict=True here
                missing_keys_overall.extend(
                    [module.prefix + k for k in result.missing_keys]
                )
                # Note: PeftBase load_state_dict removes handled keys from remaining_state_dict
                loaded_module_prefixes.add(module.prefix)
                # Remove keys associated with this module from the 'unexpected' list
                module_keys_in_sd = {
                    k for k in state_dict if k.startswith(module.prefix)
                }
                unexpected_keys_overall = [
                    k for k in unexpected_keys_overall if k not in module_keys_in_sd
                ]

            except Exception as e:
                print(
                    f"Error loading state_dict into real module {name} (prefix {module.prefix}): {e}"
                )
                # Treat keys for this module as unexpected if loading failed catastrophically
                # (PeftBase load_state_dict might have already popped keys before failing)

        # 2. Process remaining keys to create dummy modules
        # Filter keys that belong to *this* wrapper's prefix but weren't loaded yet
        potential_dummy_keys = {
            k: v for k, v in remaining_state_dict.items() if k.startswith(self.prefix)
        }

        processed_dummy_prefixes = set()

        # Group remaining keys by their inferred PEFT module prefix
        keys_by_prefix = {}
        suffixes_to_strip = [  # Common PEFT parameter suffixes
            "lora_down.weight",
            "lora_up.weight",
            "alpha",
            "dora_scale",
            "hada_w1_a",
            "hada_w1_b",
            "hada_w2_a",
            "hada_w2_b",
        ]
        for key in list(potential_dummy_keys.keys()):  # Iterate over copy of keys
            inferred_prefix = key
            for suffix in suffixes_to_strip:
                if key.endswith("." + suffix):
                    inferred_prefix = (
                        key[: -(len(suffix) + 1)] + "."
                    )  # Include trailing dot
                    break
            # else: # If no suffix matched, the key itself might be the prefix (unlikely but possible)
            #    print(f"Warning: Could not strip known suffix from key '{key}', assuming prefix is the key itself.")

            if inferred_prefix not in keys_by_prefix:
                keys_by_prefix[inferred_prefix] = {}
            keys_by_prefix[inferred_prefix][key] = potential_dummy_keys[key]

        # Now iterate through the inferred prefixes and create dummies if needed
        for peft_prefix, keys_for_this_prefix in keys_by_prefix.items():
            if (
                peft_prefix in loaded_module_prefixes
                or peft_prefix in processed_dummy_prefixes
            ):
                # Already handled by a real module or a previous dummy pass
                # Remove keys from unexpected list
                unexpected_keys_overall = [
                    k for k in unexpected_keys_overall if not k.startswith(peft_prefix)
                ]
                continue

            # Determine the original layer name corresponding to this prefix
            # Example: peft_prefix = "lora_unet_down_blocks_0_resnets_0_conv1."
            # -> module_name = "down_blocks_0_resnets_0_conv1"
            relative_name_part = peft_prefix.removeprefix(
                self.prefix + "_"
            ).removesuffix(".")
            # Need to revert the underscore sanitization done during creation
            original_layer_name = relative_name_part.replace(
                "_", "."
            )  # Best guess reverse mapping
            # TODO: This reverse mapping might be imperfect if original names had underscores.
            # A more robust way would be to store the mapping during init, but adds complexity.

            # Create the Dummy instance using default rank/alpha
            dummy_args = [peft_prefix, None, self.default_rank, self.default_alpha]
            dummy_kwargs = self.global_additional_kwargs.copy()
            try:
                print(
                    f"[LoRA Load] Creating dummy for unloaded PEFT prefix: {peft_prefix} (inferred original name: {original_layer_name})"
                )
                dummy_module = self.dummy_klass(*dummy_args, **dummy_kwargs)

                # Load the relevant keys into the dummy
                # Pass a dict containing *only* keys for this prefix to the dummy's load_state_dict
                dummy_load_dict = keys_for_this_prefix.copy()
                result = dummy_module.load_state_dict(
                    dummy_load_dict, strict=True
                )  # Use strict=True

                # Add the dummy to our managed modules, using the inferred original name as key
                self.lora_modules[original_layer_name] = dummy_module
                processed_dummy_prefixes.add(peft_prefix)
                loaded_module_prefixes.add(peft_prefix)  # Mark as loaded

                # Remove keys loaded by the dummy from the unexpected list
                keys_actually_loaded_by_dummy = list(
                    keys_for_this_prefix.keys()
                )  # Assume dummy loaded all passed keys
                unexpected_keys_overall = [
                    k
                    for k in unexpected_keys_overall
                    if k not in keys_actually_loaded_by_dummy
                ]
                # Report any issues from the dummy load (should be rare)
                if result.missing_keys:
                    print(
                        f"Warning: Dummy module {peft_prefix} reported missing keys during load: {result.missing_keys}"
                    )
                if result.unexpected_keys:
                    print(
                        f"Warning: Dummy module {peft_prefix} reported unexpected keys during load: {result.unexpected_keys}"
                    )

            except Exception as e:
                print(f"Error creating or loading dummy for prefix {peft_prefix}: {e}")
                # Keys remain unexpected if dummy creation failed

        # Final check for any remaining unexpected keys (should be few or none)
        final_unexpected = [
            k for k in unexpected_keys_overall if k.startswith(self.prefix)
        ]
        if final_unexpected:
            print(
                f"[LoRA WARNING] Unexpected keys found in state_dict for prefix '{self.prefix}' after loading:"
            )
            for k in final_unexpected:
                print(f"  - {k}")
        if missing_keys_overall:
            print(
                f"[LoRA WARNING] Missing keys for prefix '{self.prefix}' during state_dict load:"
            )
            for k in missing_keys_overall:
                print(f"  - {k}")

    def state_dict(self) -> dict:
        """Returns the state dict containing keys from all managed modules (real and dummy)."""
        state_dict = {}
        for module in self.lora_modules.values():
            # PeftBase.state_dict already adds the correct module prefix
            module_sd = module.state_dict()
            state_dict.update(module_sd)
        return state_dict

    def parameters(self) -> list[Parameter]:
        """Returns a list of trainable parameters from all *real* PEFT modules."""
        parameters = []
        for module in self.lora_modules.values():
            # Include parameters only from real (non-dummy) modules that are initialized
            if not isinstance(module, self.dummy_klass) and module._initialized:
                parameters.extend(list(module.parameters()))
        return parameters

    def requires_grad_(self, requires_grad: bool):
        """Sets requires_grad for all parameters of *real* PEFT modules."""
        count = 0
        for module in self.lora_modules.values():
            if not isinstance(module, self.dummy_klass) and module._initialized:
                module.requires_grad_(requires_grad)
                # Count parameters affected for feedback
                for _ in module.parameters():
                    count += 1
        # print(f"Set requires_grad={requires_grad} for {count} parameters in real PEFT modules.")

    def to(
        self, device: torch.device = None, dtype: torch.dtype = None
    ) -> "LoRAModuleWrapper":
        """Moves all managed modules (real and dummy) to the specified device/dtype."""
        for name, module in self.lora_modules.items():
            try:
                # `to()` is generally safe even if module has no parameters (like dummies before load)
                module.to(device=device, dtype=dtype)
            except Exception as e:
                print(
                    f"Error moving module {name} (prefix {module.prefix}) to {device}/{dtype}: {e}"
                )
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
            # Only hook real modules that have an associated original module
            if (
                not isinstance(module, self.dummy_klass)
                and module._orig_module is not None
            ):
                try:
                    module.hook_to_module()
                    hook_count += 1
                except Exception as e:
                    print(
                        f"Error applying hook for PEFT module {name} (prefix {module.prefix}): {e}"
                    )
        # print(f"Applied forward hooks for {hook_count} real PEFT modules.")

    def remove_hook_from_module(self):
        """Removes the forward hook from the original modules."""
        if self.orig_module is None:
            # print("Warning: Cannot remove hooks without the original root module (or hooks were never applied).")
            return
        remove_count = 0
        for name, module in self.lora_modules.items():
            if (
                not isinstance(module, self.dummy_klass)
                and module._orig_module is not None
            ):
                try:
                    module.remove_hook_from_module()
                    remove_count += 1
                except Exception as e:
                    print(
                        f"Error removing hook for PEFT module {name} (prefix {module.prefix}): {e}"
                    )
        # print(f"Removed forward hooks from {remove_count} real PEFT modules.")

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
            ):
                try:
                    module.apply_to_module()  # Implementation is in PeftBase subclasses (TODO)
                    apply_count += 1
                except NotImplementedError:
                    print(
                        f"Warning: apply_to_module not implemented for {type(module).__name__} ({name})."
                    )
                except Exception as e:
                    print(
                        f"Error applying weights for PEFT module {name} (prefix {module.prefix}): {e}"
                    )
        # print(f"Applied weights for {apply_count} real PEFT modules (where implemented).")

    def extract_from_module(self, base_module: nn.Module):
        """
        Extracts PEFT weights by comparing the current original module with a base module.
        (Requires implementation in LoRAModule, DoRAModule, etc.)
        """
        if self.orig_module is None:
            print(
                "Warning: Cannot extract weights without the current original root module."
            )
            return
        extract_count = 0
        for name, module in self.lora_modules.items():
            if (
                not isinstance(module, self.dummy_klass)
                and module._orig_module is not None
            ):
                try:
                    # Find the corresponding submodule in the provided base model
                    corresponding_base_submodule = base_module.get_submodule(name)
                    module.extract_from_module(
                        corresponding_base_submodule
                    )  # Implementation in PeftBase subclasses (TODO)
                    extract_count += 1
                except AttributeError:
                    print(
                        f"Warning: Could not find base submodule '{name}' during extraction."
                    )
                except NotImplementedError:
                    print(
                        f"Warning: extract_from_module not implemented for {type(module).__name__} ({name})."
                    )
                except Exception as e:
                    print(
                        f"Error extracting weights for PEFT module {name} (prefix {module.prefix}): {e}"
                    )
        # print(f"Extracted weights for {extract_count} real PEFT modules (where implemented).")

    def prune(self):
        """Removes all dummy modules from management."""
        initial_count = len(self.lora_modules)
        self.lora_modules = {
            k: v
            for (k, v) in self.lora_modules.items()
            if not isinstance(v, self.dummy_klass)
        }
        pruned_count = initial_count - len(self.lora_modules)
        # print(f"Pruned {pruned_count} dummy modules. Remaining modules: {len(self.lora_modules)}")

    def set_dropout(self, dropout_probability: float):
        """Sets the dropout probability for all *real* PEFT modules."""
        if not 0 <= dropout_probability <= 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        count = 0
        for module in self.lora_modules.values():
            if not isinstance(module, self.dummy_klass):
                # Check if the module has a dropout attribute and it's an nn.Dropout instance
                if hasattr(module, "dropout") and isinstance(
                    getattr(module, "dropout", None), nn.Dropout
                ):
                    module.dropout.p = dropout_probability
                    count += 1
        # print(f"Set dropout probability to {dropout_probability} for {count} real PEFT modules.")

    # --- Block Key Generation Logic (from modified, seems useful) ---
    @staticmethod
    def get_block_id_from_key_prefix(lora_module_prefix_with_dot: str) -> str:
        """
        Determines the Block ID (IN00-OUT11, M00, BASE) from the full PEFT module prefix.
        Used for organizing keys. Returns 'BASE' if not specifically mapped.
        """
        # Remove the trailing dot for matching
        lora_module_prefix = lora_module_prefix_with_dot.removesuffix(".")

        # Sort mapping keys by length descending for most specific match first
        # Use the PEFT prefix directly which should include 'lora_unet_' or 'lora_te_' etc.
        sorted_map_prefixes = sorted(KEY_TO_BLOCK_MAPPING.keys(), key=len, reverse=True)

        for map_prefix in sorted_map_prefixes:
            # Check if the module prefix starts with the mapping prefix
            # Example: lora_module_prefix = "lora_unet_down_blocks_0_resnets_0_conv1"
            # map_prefix = "lora_unet_down_blocks_0_resnets_0" -> Match!
            if lora_module_prefix.startswith(map_prefix):
                return KEY_TO_BLOCK_MAPPING[map_prefix]

        # Fallback if no specific UNet/TE block matched
        # print(f"Debug: PEFT prefix '{lora_module_prefix}' not mapped to specific block. Assigning BASE.")
        return "BASE"

    def generate_keys_by_block_file(
        self, output_filename="unetKeysByBlock_generated.txt"
    ):
        """Generates a text file with all managed PEFT keys organized by UNet block."""
        if not self.lora_modules:
            print("Warning: No LoRA/DoRA/LoHa modules found. Key file not generated.")
            return

        print(f"Generating key file by block: {output_filename}")
        from collections import defaultdict

        blocks_data: dict[str, list[str]] = defaultdict(list)
        all_found_block_ids = set()

        # Iterate over managed modules (real and dummies)
        for module_instance in self.lora_modules.values():
            # Get the correct prefix stored in the PEFT module instance
            module_prefix_with_dot = module_instance.prefix
            # Map the prefix to a block ID (IN00, M00, BASE, etc.)
            block_id = self.get_block_id_from_key_prefix(module_prefix_with_dot)
            all_found_block_ids.add(block_id)

            # Get keys for this module using its state_dict()
            try:
                module_state_dict = module_instance.state_dict()
                # Add keys (which already include the full prefix) to the block's list
                for key in module_state_dict.keys():
                    blocks_data[block_id].append(key)
            except Exception as e:
                print(
                    f"Error getting state_dict for module with prefix {module_prefix_with_dot}: {e}"
                )

        # Ensure all standard block IDs are present in the output, even if empty
        final_block_ids = set(BLOCKID26) | all_found_block_ids
        # Sort IDs: BASE first, then numerically/alphabetically
        sorted_block_ids = sorted(
            list(final_block_ids), key=lambda x: ("0" if x == "BASE" else "1") + x
        )

        # Format and write the file
        output_lines = []
        for block_id in sorted_block_ids:
            output_lines.append(f"=== {block_id} ===")
            # Sort keys within each block alphabetically
            keys_in_block = sorted(blocks_data.get(block_id, []))  # Use .get for safety
            if keys_in_block:
                output_lines.extend(keys_in_block)
            else:
                output_lines.append("(Empty)")  # Indicate empty blocks
            output_lines.append("")  # Blank line between blocks

        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines).strip())
            print(f"Successfully generated key file '{output_filename}'")
        except Exception as e:
            print(f"Error writing key file '{output_filename}': {e}")


# --- Helper Function (from modified, seems useful) ---
def make_layer_patterns(
    layers: list[str], rank: int | None = None, alpha: float | None = None
) -> dict[str, dict[str, int | float]]:
    """Creates a pattern dictionary for RuleSet from a list of layer names."""
    config = {}
    for layer in layers:
        entry = {}
        if rank is not None:
            entry["rank"] = rank
        if alpha is not None:
            entry["alpha"] = alpha
        if entry:  # Only add if rank or alpha was specified
            config[layer] = entry
    return config


# FIM ALTERAÇÃO: Fim do LoRAModuleWrapper da versão modificada
