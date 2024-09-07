import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import math

from utils import point_transforms



class HeadPoseRegressor(object):
    def __init__(self, model_path, use_gpu) -> None:
        super(HeadPoseRegressor, self).__init__()
        self.net = models.resnet18(num_classes=9)
        self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.net.eval()

        if use_gpu:
            self.net.cuda()

    @torch.no_grad()
    def forward(self, x, return_srt=False):
        if x.shape[2] != 128 or x.shape[3] != 128:
            x = F.interpolate(x, size=(128, 128), mode='bilinear')

        scale, rotation, translation = self.net(x).split([3, 3, 3], dim=1)
        thetas = point_transforms.get_transform_matrix(scale, rotation, translation)

        if return_srt:
            return thetas, scale, rotation, translation
        else:
            return thetas