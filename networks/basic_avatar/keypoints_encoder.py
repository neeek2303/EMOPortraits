import torch
from torch import nn
from torch.nn import functional as F
import math

from ..common import layers, params_decomposer



class KeypointsEncoder(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_harmonics: int,
                 num_channels: int,
                 num_layers: int,
                 output_channels: int,
                 output_size: int) -> None:
        super(KeypointsEncoder, self).__init__()
        self.output_channels = output_channels
        self.output_size = output_size

        self.register_buffer(
            'frequency_bands', 
            2.0 ** torch.linspace(
                0.0, 
                num_harmonics - 1, 
                num_harmonics).view(1, 1, 1, num_harmonics),
            persistent=False)

        layers_ = [nn.Linear(num_inputs * (2 + 2 * 2 * num_harmonics), num_channels)]

        for i in range(max(num_layers - 2, 0)):
            layers_ += [
                nn.ReLU(inplace=True),
                nn.Linear(num_channels, num_channels)]

        layers_ += [
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, output_channels * output_size**2, bias=False)]

        self.net = nn.Sequential(*layers_)

    def forward(self, kp):
        kp = kp[..., None]

        # Harmonic encoding: B x 68 x 2 x 1 + 2 * num_harmonics
        z = torch.cat([kp, torch.sin(kp * self.frequency_bands), torch.cos(kp * self.frequency_bands)], dim=3)
        z = z.view(z.shape[0], -1)

        z = self.net(z)
        z = z.view(z.shape[0], self.output_channels, self.output_size, self.output_size)

        return z