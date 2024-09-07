import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import models
from torch.cuda import amp
from argparse import ArgumentParser
import math
from . import GridSample
from . import utils
import numpy as np
import copy
from scipy import linalg
import itertools
from .utils import ProjectorConv, ProjectorNorm, assign_adaptive_conv_params,assign_adaptive_norm_params
from networks import basic_avatar, volumetric_avatar
from typing import List, Union

from dataclasses import dataclass, field


class VPN_ResBlocks(nn.Module):
    @dataclass
    class Config:
        num_gpus: int
        norm_layer_type: str
        input_channels: int
        num_blocks: int
        activation_type: str
        conv_layer_type: str = 'conv_3d'
        channels: list = field(default_factory=list) 
      

    def __init__(self, cfg: Config):
        super(VPN_ResBlocks, self).__init__()
        self.cfg = cfg
        self.net = volumetric_avatar.ResBlocks3d(
                num_gpus=self.cfg.num_gpus,
                norm_layer_type=self.cfg.norm_layer_type,
                input_channels=self.cfg.input_channels,
                num_blocks=self.cfg.num_blocks,
                activation_type=self.cfg.activation_type,
                conv_layer_type='conv_3d',
                channels=self.cfg.channels,
            )
        
    def forward(self, x):

        return self.net(x)