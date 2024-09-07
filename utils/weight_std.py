"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
"""
import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from torch.nn import Module
from torch import nn



def apply_weight_std(module, name='weight', apply_to=['conv2d'], n_power_iterations=1, eps=1e-12):
    # Apply only to modules in apply_to list
    module_name = module.__class__.__name__.lower()
    if module_name not in apply_to:
        return module

    WeightStd.apply(module, name, eps)

    return module

def remove_weight_std(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightStd) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break

    return module


class WeightStd:
    name: str
    eps: float

    def __init__(self, name: str = 'weight', eps: float = 1e-5) -> None:
        self.name = name
        self.eps = eps

    def compute_weight(self, module: Module) -> torch.Tensor:
        weight_orig = getattr(module, self.name + '_orig')
        weight = weight_orig.view(weight_orig.shape[0], -1)

        mu = weight.mean(dim=1, keepdim=True)
        sigma = weight.std(dim=1, keepdim=True)

        weight = (weight - mu) / (sigma + self.eps)

        weight = weight.view(weight_orig.shape)
        
        return weight

    def remove(self, module: Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module)
        delattr(module, self.name)
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module: Module, name: str, eps: float) -> 'WeightStd':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightStd) and hook.name == name:
                raise RuntimeError("Cannot register two weight_std hooks on "
                                   "the same parameter {}".format(name))

        fn = WeightStd(name, eps)
        weight = module._parameters[name]

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)

        module.register_forward_pre_hook(fn)
        return fn


T_module = TypeVar('T_module', bound=Module)

def weight_std(module: T_module,
               name: str = 'weight',
               eps: float = 1e-12) -> T_module:
    WeightStd.apply(module, name, eps)
    return module