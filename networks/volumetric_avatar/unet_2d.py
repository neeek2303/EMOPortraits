import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from . import utils

norm_layers = {
    'in': nn.InstanceNorm2d,
    'bn': nn.SyncBatchNorm,
    'sync_bn': nn.SyncBatchNorm,
    'gn': lambda num_features, affine=True: nn.GroupNorm(num_groups=32, num_channels=num_features, affine=affine),

}

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels,  mid_channels=None, norm='bn'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        norm_layer = norm_layers[norm]

        # self.double_conv = utils.blocks['res'](
        #             in_channels=in_channels,
        #             out_channels=out_channels,
        #             stride=1,
        #             norm_layer_type=norm,
        #             activation_type='relu')


        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm=norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm='bn'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base=64, max_ch=1024, bilinear=True, norm='bn'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base, norm=norm)
        self.down1 = Down(base, base*2, norm)
        self.down2 = Down(base*2, base*4, norm)
        self.down3 = Down(base*4, min(base*8, max_ch), norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(min(base*8, max_ch), min(base*16,max_ch) // factor, norm)
        self.up1 = Up(min(base*16, max_ch), min(base*8, max_ch) // factor, bilinear, norm)
        self.up2 = Up(min(base*8, max_ch), base*4 // factor, bilinear, norm)
        # self.up3 = Up(base*4, base*2 // factor, bilinear, norm)
        # self.up4 = Up(base*2, base, bilinear, norm)
        # self.outc = OutConv(base, n_classes)

        self.up3 = Up(base*4, base*4 // factor, bilinear, norm)
        self.up4 = Up(base*3, base*2, bilinear, norm)
        self.outc = OutConv(base*2, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits