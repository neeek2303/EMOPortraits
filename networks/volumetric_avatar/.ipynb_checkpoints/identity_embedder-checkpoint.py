import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.cuda import amp
from torchvision import models
import itertools
# import apex
from . import utils
from dataclasses import dataclass

class IdtEmbed(nn.Module):
    @dataclass
    class Config:
        idt_backbone: str
        num_source_frames: int
        idt_output_size: int
        idt_output_channels: int
        num_gpus: int
        norm_layer_type: str
        idt_image_size: int


    def __init__(self, cfg):
        super(IdtEmbed, self).__init__()
        self.cfg = cfg
        EXPANSION = 1 if self.cfg.idt_backbone == 'resnet18' else 4
        self.num_source_frames = self.cfg.num_source_frames # number of source imgs per identity
        self.net = getattr(models, self.cfg.idt_backbone)(pretrained=True)
        self.idt_image_size = self.cfg.idt_image_size
        # Patch backbone according to args
        self.net.avgpool = nn.AdaptiveAvgPool2d(self.cfg.idt_output_size)

        num_outputs = self.cfg.idt_output_channels

        self.net.fc = nn.Conv2d(
            in_channels=512 * EXPANSION,
            out_channels=num_outputs,
            kernel_size=1, 
            bias=False)

        if self.cfg.norm_layer_type=='bn':
            pass
            # if self.cfg.num_gpus > 1:
            # #     # self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            #     self.net = apex.parallel.convert_syncbn_model(self.net)
        elif self.cfg.norm_layer_type=='in':
            self.net = utils.replace_bn_to_in(self.net, 'IdtEmbed')
        elif self.cfg.norm_layer_type=='gn':
            self.net = utils.replace_bn_to_gn(self.net, 'IdtEmbed')
        elif self.cfg.norm_layer_type == 'bcn':
            self.net = utils.replace_bn_to_bcn(self.net, 'IdtEmbed')
        else:
            raise ValueError('wrong norm type')

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std',  torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

    def _forward_impl(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.fc(x)
        x = self.net.avgpool(x)

        return x

    def forward(self, source_img):
        idt_embed = self.forward_image(source_img)

        return idt_embed

    def forward_image(self, source_img):
        source_img = F.interpolate(source_img, size=(self.idt_image_size, self.idt_image_size), mode='bilinear')
        n = self.num_source_frames
        b = source_img.shape[0] // n

        inputs = (source_img - self.mean) / self.std
        idt_embed_tensor = self._forward_impl(inputs)
        idt_embed_tensor = idt_embed_tensor.view(b, n, *idt_embed_tensor.shape[1:]).mean(1)

        return idt_embed_tensor




