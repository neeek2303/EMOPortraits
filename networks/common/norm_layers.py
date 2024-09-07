import torch
from torch import nn
# import apex



def init_parameters(self, num_features):
    self.weight = nn.Parameter(torch.ones(num_features))
    self.bias = nn.Parameter(torch.zeros(num_features))
    
    # These tensors are assigned externally
    self.ada_weight = None
    self.ada_bias = None

def common_forward(x, weight, bias):
    B = weight.shape[0]
    T = x.shape[0] // B

    x = x.view(B, T, *x.shape[1:])

    if len(weight.shape) == 2:
        # Broadcast weight and bias accross T and spatial size of outputs
        if len(x.shape) == 5:
            x = x * weight[:, None, :, None, None] + bias[:, None, :, None, None]
        elif len(x.shape) == 6:
            x = x * weight[:, None, :, None, None, None] + bias[:, None, :, None, None, None]
    else:
        x = x * weight[:, None] + bias[:, None]

    x = x.view(B*T, *x.shape[2:])

    return x


class AdaptiveInstanceNorm(nn.modules.instancenorm._InstanceNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(AdaptiveInstanceNorm, self).__init__(
            num_features, eps, momentum, False, track_running_stats)
        init_parameters(self, num_features)
        
    def forward(self, x):
        x = super(AdaptiveInstanceNorm, self).forward(x)
        x = common_forward(x, self.ada_weight, self.ada_bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class AdaptiveBatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(AdaptiveBatchNorm, self).__init__(
            num_features, eps, momentum, False, track_running_stats)
        init_parameters(self, num_features)
        
    def forward(self, x):
        x = super(AdaptiveBatchNorm, self).forward(x)
        x = common_forward(x, self.ada_weight, self.ada_bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


# class AdaptiveSyncBatchNorm(apex.parallel.SyncBatchNorm):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):
#         super(AdaptiveSyncBatchNorm, self).__init__(
#             num_features, eps, momentum, False, track_running_stats)
#         init_parameters(self, num_features)
        
#     def forward(self, x):
#         x = super(AdaptiveSyncBatchNorm, self).forward(x)
#         x = common_forward(x, self.ada_weight, self.ada_bias)

#         return x

#     def _check_input_dim(self, input):
#         pass

#     def extra_repr(self):
#         return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
#                'track_running_stats={track_running_stats}'.format(**self.__dict__)


class AdaptiveGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_features, eps=1e-5, affine=True):
        super(AdaptiveGroupNorm, self).__init__(num_groups, num_features, eps, False)
        self.num_features = num_features
        init_parameters(self, num_features)
        
    def forward(self, x):
        x = super(AdaptiveGroupNorm, self).forward(x)
        x = common_forward(x, self.ada_weight, self.ada_bias)

        return x

    def _check_input_dim(self, input):
        pass

    def extra_repr(self) -> str:
        return '{num_groups}, {num_features}, eps={eps}, ' \
            'affine=True'.format(**self.__dict__)