"""
References:
1) the official LoRA implementation released by Microsoft:
https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
"""

import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn


class LoRAParametrization(nn.Module):
    def __init__(self, fan_in, fan_out, fan_in_fan_out=True, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        super().__init__()

        self.lora_A = nn.Parameter(torch.zeros(rank, fan_in))
        self.lora_B = nn.Parameter(torch.zeros(fan_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(fan_in, dtype=self.lora_A.dtype))
        self.forward_fn = self.lora_forward
        # (W + BA)x : If W is stored as (fan_in, fan_out), then we need to transpose BA to match W
        self.transpose = (lambda x: x.transpose(0, 1)) if fan_in_fan_out else (lambda x: x)

    def _dropout(self, A):
        # to mimic the original implementation: (A * dropout_mask) @ x = A @ dropout(x)
        return A * self.lora_dropout(self.lora_dropout_mask)

    def lora_forward(self, X):
        return X + self.transpose(torch.mm(self.lora_B, self.dropout_fn(self.lora_A))).view(X.shape) * self.scaling

    def forward(self, X):
        return self.forward_fn(X)

    def disable_lora(self):
        self.forward_fn = lambda x: x

    def enable_lora(self):
        self.forward_fn = self.lora_forward

    @classmethod
    def from_linear(cls, layer, rank=4):
        fan_out, fan_in = layer.weight.shape
        return cls(fan_in, fan_out, fan_in_fan_out=False, rank=rank)

    @classmethod
    def from_conv2d(cls, layer, rank=4):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(fan_in, fan_out, fan_in_fan_out=False, rank=rank)

    @classmethod
    def from_embedding(cls, layer, rank=4):
        fan_in, fan_out = layer.weight.shape
        return cls(fan_in, fan_out, fan_in_fan_out=True, rank=rank)


default_lora_config = {  # specify which layers to add lora to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=4),
    },
}


def apply_lora(layer, register=True, merge=False, lora_config=default_lora_config):
    """add lora parametrization to a layer, designed to be used with model.apply"""
    if register:
        if type(layer) in lora_config:
            for attr_name, parametrization in lora_config[type(layer)].items():
                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
    else:  # this will remove all parametrizations, use with caution
        if hasattr(layer, "parametrizations"):
            for attr_name in layer.parametrizations.keys():
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)


def add_lora(model, lora_config=default_lora_config):
    """add lora parametrization to all layers in a model. Calling it twice will add lora twice"""
    model.apply(partial(apply_lora, lora_config=lora_config))


def merge_lora(model):
    """merge lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=True))


def remove_lora(model):
    """remove lora parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_lora, register=False, merge=False))


def apply_to_lora(fn):
    """apply a function to LoRAParametrization layers, designed to be used with model.apply"""

    def apply_fn(layer):
        if isinstance(layer, LoRAParametrization):
            fn(layer)

    return apply_fn


enable_lora = lambda model: model.apply(apply_to_lora(lambda x: x.enable_lora()))
disable_lora = lambda model: model.apply(apply_to_lora(lambda x: x.disable_lora()))


# ------------------- helper function for inferencing with multiple lora -------------------


def prepare_for_multiple_lora(lora_layer):
    lora_layer.lora_As = []
    lora_layer.lora_Bs = []


def _append_lora(lora_layer):
    lora_layer.lora_As.append(torch.nn.Parameter(lora_layer.lora_A.clone()))
    lora_layer.lora_Bs.append(torch.nn.Parameter(lora_layer.lora_B.clone()))


def load_multiple_lora(model, lora_state_dicts):
    model.apply(apply_to_lora(prepare_for_multiple_lora))
    for state_dict in lora_state_dicts:
        _ = model.load_state_dict(state_dict, strict=False)
        model.apply(apply_to_lora(_append_lora))
    return model


def _select_lora(lora_layer, index):
    lora_layer.lora_A = lora_layer.lora_As[index]
    lora_layer.lora_B = lora_layer.lora_Bs[index]


def select_lora(model, index):
    model.apply(apply_to_lora(lambda x: _select_lora(x, index)))
    return model
