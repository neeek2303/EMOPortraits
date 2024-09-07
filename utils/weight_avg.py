import torch
from torch import nn



class WeightAveragingHook(object):
    # Mode can be either "running_average" with momentum
    # or "average" for direct averaging
    def __init__(self, name='weight', mode='running_average', momentum=0.9999):
        self.name = name
        self.mode = mode
        self.momentum = momentum # running average parameter
        self.num_iter = 1 # average parameter

    def update_param(self, module):
        # Only update average values
        param = getattr(module, self.name)
        param_avg = getattr(module, self.name + '_avg')
        with torch.no_grad():
            if self.mode == 'running_average':
                param_avg.data = param_avg.data * self.momentum + param.data * (1 - self.momentum)
            elif self.mode == 'average':
                param_avg.data = (param_avg.data * self.num_iter + param.data) / (self.num_iter + 1)
                self.num_iter += 1

    def remove(self, module):
        param_avg = getattr(module, self.name + '_avg')
        delattr(module, self.name)
        delattr(module, self.name + '_avg')
        module.register_parameter(self.name, nn.Parameter(param_avg))

    def __call__(self, module, grad_input, grad_output):
        if module.training: 
            self.update_param(module)

    @staticmethod
    def apply(module, name, mode, momentum):
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, WeightAveragingHook) and hook.name == name:
                raise RuntimeError("Cannot register two weight_averaging hooks on "
                                   "the same parameter {}".format(name))
                
        fn = WeightAveragingHook(name, mode, momentum)
        
        if name in module._parameters:
            param = module._parameters[name].data
        else:
            param = getattr(module, name)

        module.register_buffer(name + '_avg', param.clone())

        module.register_backward_hook(fn)
        
        return fn


class WeightAveragingPreHook(object):
    # Mode can be either "running_average" with momentum
    # or "average" for direct averaging
    def __init__(self, name='weight'):
        self.name = name
        self.spectral_norm = True
        self.enable = False

    def __call__(self, module, inputs):
        if self.enable or not module.training:
            setattr(module, self.name, getattr(module, self.name + '_avg'))

        elif not self.spectral_norm:
            setattr(module, self.name, getattr(module, self.name + '_orig') + 0) # +0 converts a parameter to a tensor with grad fn

    @staticmethod
    def apply(module, name):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightAveragingPreHook) and hook.name == name:
                raise RuntimeError("Cannot register two weight_averaging hooks on "
                                   "the same parameter {}".format(name))
                
        fn = WeightAveragingPreHook(name)

        if not hasattr(module, name + '_orig'):
            param = module._parameters[name]

            delattr(module, name)
            module.register_parameter(name + '_orig', param)
            setattr(module, name, param.data)

            fn.spectral_norm = False

        module.register_forward_pre_hook(fn)
        
        return fn


def weight_averaging(module, names=['weight', 'bias'], mode='running_average', momentum=0.9999):
    for name in names:
        if hasattr(module, name) and getattr(module, name) is not None:
            WeightAveragingHook.apply(module, name, mode, momentum)
            WeightAveragingPreHook.apply(module, name)

    return module

def remove_weight_averaging(module, names=['weight', 'bias']):
    for name in names:               
        for k, hook in module._backward_hooks.items():
            if isinstance(hook, WeightAveragingHook) and hook.name == name:
                hook.remove(module)
                del module._backward_hooks[k]
                break

        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightAveragingPreHook) and hook.name == name:
                hook.remove(module)
                del module._forward_pre_hooks[k]
                break

    return module
