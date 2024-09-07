import torch
from torch import nn



class StatsCalcHook(object):
    def __init__(self):
        self.num_iter = 0
    
    def update_stats(self, module):
        for stats_name in ['mean', 'var']:
            batch_stats = getattr(module, f'running_{stats_name}')
            accum_stats = getattr(module, f'accumulated_{stats_name}')
            accum_stats = accum_stats + batch_stats
            setattr(module, f'accumulated_{stats_name}', accum_stats)
        
        self.num_iter += 1

    def remove(self, module):
        for stats_name in ['mean', 'var']:
            accum_stats = getattr(module, f'accumulated_{stats_name}') / self.num_iter
            delattr(module, f'accumulated_{stats_name}')
            getattr(module, f'running_{stats_name}').data = accum_stats

    def __call__(self, module, inputs, outputs):
        self.update_stats(module)

    @staticmethod
    def apply(module):
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, StatsCalcHook):
                raise RuntimeError("Cannot register two calc_stats hooks on "
                                   "the same module")
                
        fn = StatsCalcHook()
        
        stats = getattr(module, 'running_mean')
        for stats_name in ['mean', 'var']:
            attr_name = f'accumulated_{stats_name}'
            if hasattr(module, attr_name): 
                delattr(module, attr_name)
            module.register_buffer(attr_name, torch.zeros_like(stats))

        module.register_forward_hook(fn)
        
        return fn


def stats_calculation(module):
    if 'BatchNorm' in module.__class__.__name__:
        module.momentum = 1.0
        StatsCalcHook.apply(module)

    return module

def remove_stats_calculation(module):
    if 'BatchNorm' in module.__class__.__name__:
        for k, hook in module._forward_hooks.items():
            if isinstance(hook, StatsCalcHook):
                hook.remove(module)
                del module._forward_hooks[k]
                return module

    return module