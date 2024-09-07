import torch
from torch import nn
from torch.nn import functional as F

from ..common import layers, params_decomposer



class MotionFieldEstimator(nn.Module):
    def __init__(self,
                 min_channels: int,
                 max_channels: int,
                 output_depth: int,
                 embed_channels: int,
                 block_type: str,
                 num_groups: int,
                 num_blocks: int,
                 num_layers: int,
                 norm_layer_type: str,
                 activation_type: str,
                 resize_depth: bool) -> None:
        super(MotionFieldEstimator, self).__init__()
        expansion_factor = 4 if block_type == 'bottleneck' else 1
        stride = 2 if resize_depth else (1, 2, 2)
        num_channels = [max_channels] + [min(min_channels * 2**i, max_channels) for i in reversed(range(num_groups))]

        if resize_depth:
            self.inputs = nn.Parameter(torch.randn(1, max_channels * expansion_factor, 4, 4, 4, requires_grad=True))
        else:
            self.inputs = nn.Parameter(torch.randn(1, max_channels * expansion_factor, output_depth, 4, 4, requires_grad=True))

        layers_ = []

        for i in range(1, num_groups + 1):
            layers_.append(nn.Upsample(scale_factor=stride))

            for j in range(num_blocks):
                layers_.append(layers.blocks[block_type](
                    in_channels=num_channels[i - 1 if j == 0 else i],
                    out_channels=num_channels[i],
                    num_layers=num_layers,
                    expansion_factor=expansion_factor,
                    kernel_size=3,
                    stride=1,
                    norm_layer_type=f'ada_{norm_layer_type}',
                    activation_type=activation_type,
                    conv_layer_type='conv_3d'))

        layers_ += [
            layers.norm_layers[f'ada_{norm_layer_type}'](num_channels[-1] * expansion_factor),
            layers.activations[activation_type](inplace=True),
            nn.Conv3d(
                in_channels=num_channels[-1] * expansion_factor,
                out_channels=3,
                kernel_size=1),
            nn.Tanh()]

        self.net = nn.Sequential(*layers_)

        self.pred_params = params_decomposer.NormParamsPredictor(self.net, embed_channels)

    def init(self):
        last_conv = list(self.net.modules())[-2]
        
        nn.init.xavier_normal_(last_conv.weight, gain=0.02)
        nn.init.zeros_(last_conv.bias)

    def forward(self, embeds):
        b = embeds.shape[0]

        params = self.pred_params(embeds)
        params_decomposer.assign_adaptive_norm_params(self.net, params)

        delta_w = self.net(self.inputs.repeat_interleave(b, dim=0))

        if not hasattr(self, 'identity_grid'):
            _, _, d, h, w = delta_w.shape
            grid_x = torch.linspace(-1, 1, w)
            grid_y = torch.linspace(-1, 1, h)
            grid_z = torch.linspace(-1, 1, d)
            w, v, u = torch.meshgrid(grid_z, grid_y, grid_x)
            self.register_buffer('identity_grid', torch.stack([u, v, w], dim=0)[None], persistent=False)

            self.identity_grid = self.identity_grid.type(delta_w.type()).to(delta_w.device)

        w = (self.identity_grid + delta_w).permute(0, 2, 3, 4, 1).clamp(-1, 1) # grid sampler expects channels last

        return w, delta_w