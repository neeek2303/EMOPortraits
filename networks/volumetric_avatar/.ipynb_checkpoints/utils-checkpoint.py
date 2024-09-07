import torch
from torch import nn
import torch.nn.functional as F
import math
import functools
from einops import rearrange, repeat
import itertools
import torchvision
from typing import List, Union
from utils import misc
from torch import nn, autograd, optim
import numpy as np
from utils import args as args_utils

MOMENTUM = 0.01
@torch.no_grad()
def grid_sampler_backward(grad_out, grid, h=None, w=None, padding_mode='zeros', align_corners=False):
    b, c = grad_out.shape[:2]
    if h is None or w is None:
        h, w = grad_out.shape[2:]
    size = torch.FloatTensor([w, h]).to(grad_out.device)
    grad_in = torch.zeros(b, c, h, w, device=grad_out.device)

    if align_corners:
        grid_ = (grid + 1) / 2 * (size - 1)
    else:
        grid_ = ((grid + 1) * size - 1) / 2

    if padding_mode == 'border':
        assert False, 'TODO'

    elif padding_mode == 'reflection':
        assert False, 'TODO'

    grid_nw = grid_.floor().long()

    grid_ne = grid_nw.clone()
    grid_ne[..., 0] += 1

    grid_sw = grid_nw.clone()
    grid_sw[..., 1] += 1

    grid_se = grid_nw.clone() + 1

    nw = (grid_se - grid_).prod(3)
    ne = (grid_ - grid_sw).abs().prod(3)
    sw = (grid_ne - grid_).abs().prod(3)
    se = (grid_ - grid_nw).prod(3)

    indices_ = torch.cat([
        (
                (
                        g[:, None, ..., 0] + g[:, None, ..., 1] * w
                ).repeat_interleave(c, dim=1)
                + torch.arange(c, device=g.device)[None, :, None, None] * (h * w)  # add channel shifts
                + torch.arange(b, device=g.device)[:, None, None, None] * (c * h * w)  # add batch size shifts
        ).view(-1)
        for g in [grid_nw, grid_ne, grid_sw, grid_se]
    ])

    masks = torch.cat([
        (
                (g[..., 0] >= 0) & (g[..., 0] < w) & (g[..., 1] >= 0) & (g[..., 1] < h)
        )[:, None].repeat_interleave(c, dim=1).view(-1)
        for g in [grid_nw, grid_ne, grid_sw, grid_se]
    ])

    values_ = torch.cat([
        (m[:, None].repeat_interleave(c, dim=1) * grad_out).view(-1)
        for m in [nw, ne, sw, se]
    ])

    indices = indices_[masks]
    values = values_[masks]

    grad_in.put_(indices, values, accumulate=True)

    return grad_in


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        batch_size, channels, in_height, in_width = inputs.size()

        out_height = in_height // self.upscale_factor
        out_width = in_width // self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, out_height, self.upscale_factor,
            out_width, self.upscale_factor)

        channels *= self.upscale_factor ** 2
        unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return unshuffle_out.view(batch_size, channels, out_height, out_width)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(AdaptiveConv, self).__init__()
        # Set options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert not bias, 'bias == True is not supported for AdaptiveConv'
        self.bias = None

        self.kernel_numel = kernel_size[0] * kernel_size[1]
        if len(kernel_size) == 3:
            self.kernel_numel *= kernel_size[2]

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.ada_weight = None  # assigned externally

        if len(kernel_size) == 2:
            self.conv_func = F.conv2d
        elif len(kernel_size) == 3:
            self.conv_func = F.conv3d

    def forward(self, inputs):
        # Cast parameters into inputs.dtype
        if inputs.type() != self.ada_weight.type():
            weight = self.ada_weight.type(inputs.type())
        else:
            weight = self.ada_weight

        # Conv is applied to the inputs grouped by t frames
        B = weight.shape[0]
        T = inputs.shape[0] // B
        assert inputs.shape[0] == B * T, 'Wrong shape of weight'

        if self.kernel_numel > 1:
            if weight.shape[0] == 1:
                # No need to iterate through batch, can apply conv to the whole batch
                outputs = self.conv_func(inputs, weight[0], None, self.stride, self.padding, self.dilation, self.groups)

            else:
                outputs = []
                for b in range(B):
                    outputs += [self.conv_func(inputs[b * T:(b + 1) * T], weight[b], None, self.stride, self.padding,
                                               self.dilation, self.groups)]
                outputs = torch.cat(outputs, 0)

        else:
            if weight.shape[0] == 1:
                if len(inputs.shape) == 5:
                    weight = weight[..., None, None, None]
                else:
                    weight = weight[..., None, None]

                outputs = self.conv_func(inputs, weight[0], None, self.stride, self.padding, self.dilation, self.groups)
            else:
                # 1x1(x1) adaptive convolution is a simple bmm
                if len(weight.shape) == 6:
                    weight = weight[..., 0, 0, 0]
                else:
                    weight = weight[..., 0, 0]

                outputs = torch.bmm(weight, inputs.view(B * T, inputs.shape[1], -1)).view(B, -1, *inputs.shape[2:])

        return outputs

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class EstBN(nn.Module):

    def __init__(self, num_features):
        super(EstBN, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        # self.register_buffer('estbn_moving_speed', torch.zeros(1))
        self.register_buffer('estbn_moving_speed', torch.tensor(0.01))

    def forward(self, inp):
        ms = self.estbn_moving_speed.item()
        if self.training:
            with torch.no_grad():
                inp_t = inp.transpose(0, 1).contiguous().view(self.num_features, -1)
                running_mean = inp_t.mean(dim=1)
                inp_t = inp_t - self.running_mean.view(-1, 1)
                running_var = torch.mean(inp_t * inp_t, dim=1)
                self.running_mean.data.mul_(1 - ms).add_(ms * running_mean.data)
                self.running_var.data.mul_(1 - ms).add_(ms * running_var.data)

        if len(inp.shape) == 4:
            out = inp - self.running_mean.view(1, -1, 1, 1)
            out = out / torch.sqrt(self.running_var + 1e-5).view(1, -1, 1, 1)
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)

        elif len(inp.shape) == 5:
            out = inp - self.running_mean.view(1, -1, 1, 1, 1)
            out = out / torch.sqrt(self.running_var + 1e-5).view(1, -1, 1, 1, 1)
            weight = self.weight.view(1, -1, 1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1, 1)
        else:
            raise ValueError(f'Wrong inp shape {inp.shape}')

        out = weight * out + bias
        return out

class BCNorm(nn.Module):

    def __init__(self,  num_groups, num_channels, eps = 1e-5, estimate=False):
        super(BCNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self._weight = nn.Parameter(torch.ones(1, num_groups, 1))
        self._bias = nn.Parameter(torch.zeros(1, num_groups, 1))
        # self.weight = nn.Parameter(torch.ones(1, num_groups, 1))
        # self.bias = nn.Parameter(torch.zeros(1, num_groups, 1))
        if estimate:
            self.bn = EstBN(num_channels)
        else:
            # self.bn = nn.BatchNorm2d(num_channels)
            self.bn = nn.SyncBatchNorm(num_channels, momentum=MOMENTUM)

    def forward(self, inp, implicit_bn=True):
        if implicit_bn:
            out = self.bn(inp)
        else:
            out = inp
        out = out.view(1, inp.size(0) * self.num_groups, -1)
        out = torch.batch_norm(out, None, None, None, None, True, 0, self.eps, True)
        out = out.view(inp.size(0), self.num_groups, -1)
        # print(self.weight.shape, out.shape)
        # out = self.weight * out + self.bias
        out = self._weight * out + self._bias
        out = out.view_as(inp)
        return out



class AdaptiveBCNorm(BCNorm):
    def __init__(self, num_groups, num_features, eps=1e-5, estimate=False):
        super(AdaptiveBCNorm, self).__init__(num_channels=num_features, eps=eps, num_groups=num_groups, estimate=estimate)
        self.num_features = num_features
        self.bn.weight = nn.Parameter(torch.ones(num_features))
        self.bn.bias = nn.Parameter(torch.zeros(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # These tensors are assigned externally
        self.ada_weight = None
        self.ada_bias = None

    def forward(self, inputs, implicit_bn=True):
        outputs = self.bn(inputs)
        B = self.ada_weight.shape[0]
        T = inputs.shape[0] // B

        outputs = outputs.view(B, T, *outputs.shape[1:])
        # Broadcast weight and bias accross T and spatial size of outputs
        if len(outputs.shape) == 5:
            outputs = outputs * self.ada_weight[:, None, :, None, None] + self.ada_bias[:, None, :, None, None]
        else:
            outputs = outputs * self.ada_weight[:, None, :, None, None, None] + self.ada_bias[:, None, :, None, None,
                                                                                None]
        outputs = outputs.view(B * T, *outputs.shape[2:])
        outputs = super(AdaptiveBCNorm, self).forward(outputs, implicit_bn=False)
        return outputs

    def _check_input_dim(self, input):
        pass

    def extra_repr(self) -> str:
        return '{num_groups}, {num_features}, eps={eps}, ' \
               'affine=True'.format(**self.__dict__)


class AdaptiveGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_features, eps=1e-5, affine=True):
        super(AdaptiveGroupNorm, self).__init__(num_groups, num_features, eps, False)
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        # These tensors are assigned externally
        self.ada_weight = None
        self.ada_bias = None

    def forward(self, inputs):
        outputs = super(AdaptiveGroupNorm, self).forward(inputs)
        B = self.ada_weight.shape[0]
        T = inputs.shape[0] // B

        outputs = outputs.view(B, T, *outputs.shape[1:])
        # Broadcast weight and bias accross T and spatial size of outputs
        if len(outputs.shape) == 5:
            outputs = outputs * self.ada_weight[:, None, :, None, None] + self.ada_bias[:, None, :, None, None]
        else:
            outputs = outputs * self.ada_weight[:, None, :, None, None, None] + self.ada_bias[:, None, :, None, None,
                                                                                None]
        outputs = outputs.view(B * T, *outputs.shape[2:])
        return outputs

    def _check_input_dim(self, input):
        pass

    def extra_repr(self) -> str:
        return '{num_groups}, {num_features}, eps={eps}, ' \
               'affine=True'.format(**self.__dict__)

class AdaptiveInstanceNorm(nn.modules.instancenorm._InstanceNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(AdaptiveInstanceNorm, self).__init__(
            num_features, eps, momentum, False, track_running_stats)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # These tensors are assigned externally
        self.ada_weight = None
        self.ada_bias = None

    def forward(self, inputs):
        outputs = super(AdaptiveInstanceNorm, self).forward(inputs)

        B = self.ada_weight.shape[0]
        T = inputs.shape[0] // B

        outputs = outputs.view(B, T, *outputs.shape[1:])

        # Broadcast weight and bias accross T and spatial size of outputs
        if len(outputs.shape) == 5:
            outputs = outputs * self.ada_weight[:, None, :, None, None] + self.ada_bias[:, None, :, None, None]
        else:
            outputs = outputs * self.ada_weight[:, None, :, None, None, None] + self.ada_bias[:, None, :, None, None,
                                                                                None]

        outputs = outputs.view(B * T, *outputs.shape[2:])

        return outputs

    def _check_input_dim(self, input):
        pass

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)



class AdaptiveBatchNorm(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(AdaptiveBatchNorm, self).__init__(
            num_features, eps, momentum, False, track_running_stats)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # These tensors are assigned externally
        self.ada_weight = None
        self.ada_bias = None

    def forward(self, inputs):
        outputs = super(AdaptiveBatchNorm, self).forward(inputs)

        B = self.ada_weight.shape[0]
        T = inputs.shape[0] // B

        outputs = outputs.view(B, T, *outputs.shape[1:])

        # Broadcast weight and bias accross T and spatial size of outputs
        if len(outputs.shape) == 5:
            outputs = outputs * self.ada_weight[:, None, :, None, None] + self.ada_bias[:, None, :, None, None]
        else:
            outputs = outputs * self.ada_weight[:, None, :, None, None, None] + self.ada_bias[:, None, :, None, None,
                                                                                None]

        outputs = outputs.view(B * T, *outputs.shape[2:])

        return outputs

    def _check_input_dim(self, input):
        pass

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class AdaptiveSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(AdaptiveSyncBatchNorm, self).__init__(
            num_features, eps, momentum, False, track_running_stats)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # These tensors are assigned externally
        self.ada_weight = None
        self.ada_bias = None

    def forward(self, inputs):
        outputs = super(AdaptiveSyncBatchNorm, self).forward(inputs)

        B = self.ada_weight.shape[0]
        T = inputs.shape[0] // B

        outputs = outputs.view(B, T, *outputs.shape[1:])

        # Broadcast weight and bias accross T and spatial size of outputs
        if len(outputs.shape) == 5:
            outputs = outputs * self.ada_weight[:, None, :, None, None] + self.ada_bias[:, None, :, None, None]
        else:
            outputs = outputs * self.ada_weight[:, None, :, None, None, None] + self.ada_bias[:, None, :, None, None,
                                                                                None]

        outputs = outputs.view(B * T, *outputs.shape[2:])

        return outputs

    def _check_input_dim(self, input):
        pass

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    input = input.permute(0, 2, 3, 1)
    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # out = out.permute(0, 2, 3, 1)
    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    # out = UpFirDn2d.apply(
    #     input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
    # )
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    return out

class Upsample_sg2(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample_sg2(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class TransformerHead(nn.Module):
    def __init__(self, num_inputs, dim, depth, heads, dim_head, mlp_dim, dropout, emb_dropout):
        super(TransformerHead, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_inputs + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, inputs):
        b, c = inputs.shape[:2]
        x = inputs.view(b, c, -1).permute(0, 2, 1)
        n = x.shape[1]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)  # concatenate zero token
        x += self.pos_embedding[:, :(n + 1)]  # add positional embeddings
        x = self.dropout(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x[:, 0]


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            conv_layer_type: str = 'conv',
            norm_layer_type: str = 'bn',
            activation_type: str = 'relu',
            resize_layer_type: str = 'none',
            efficient_upsampling: bool = False,  # place upsampling layer before the second convolution
            return_feats: bool = False,  # return feats after the first convolution,
    ):
        """This is a base module for residual blocks"""
        super(ResBlock, self).__init__()
        # Initialize layers in the block
        self.return_feats = return_feats

        m_bias= False
        if resize_layer_type in ['nearest', 'bilinear', 'blur']:
            self.upsample = lambda inputs: F.interpolate(inputs, scale_factor=stride, mode=resize_layer_type)
            self.efficient_upsampling = efficient_upsampling
            if resize_layer_type=='blur':
                self.upsample = Upsample_sg2(kernel=[1, 3, 3, 1])

        downsample = resize_layer_type in downsampling_layers and stride > 1
        if downsample:
            downsampling_layer = downsampling_layers[resize_layer_type]

        normalize = norm_layer_type != 'none'
        if normalize:
            norm_layer = norm_layers[norm_layer_type]

        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]

        if '3d' in conv_layer_type:
            num_kernel_dims = 3
        else:
            num_kernel_dims = 2

        ### Initialize the layers of the first half of the block ###
        layers = []

        if normalize:
            layers += [norm_layer(in_channels)]

        layers += [
            activation(inplace=True),
            conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size,) * num_kernel_dims,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=m_bias)]

        if normalize:
            layers += [norm_layer(out_channels)]

        layers += [activation(inplace=True)]

        self.block_feats = nn.Sequential(*layers)

        layers = [
            conv_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size,) * num_kernel_dims,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=m_bias)]

        if downsample:
            layers += [downsampling_layer(stride)]

        self.block = nn.Sequential(*layers)

        ### Initialize a skip connection block, if needed ###
        if in_channels != out_channels or downsample:
            layers = []

            if in_channels != out_channels:
                layers += [conv_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1,) * num_kernel_dims,
                    bias=m_bias)]

            if downsample:
                layers += [downsampling_layer(stride)]

            self.skip = nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = inputs

        if hasattr(self, 'upsample') and not self.efficient_upsampling:
            outputs = self.upsample(inputs)

        feats = self.block_feats(outputs)
        outputs = feats

        if hasattr(self, 'upsample') and self.efficient_upsampling:
            outputs = self.upsample(feats)

        outputs_main = self.block(outputs)

        outputs_skip = inputs

        if hasattr(self, 'upsample'):
            outputs_skip = self.upsample(inputs)

        if hasattr(self, 'skip'):
            outputs_skip = self.skip(outputs_skip)

        outputs = outputs_main + outputs_skip

        if self.return_feats:
            return outputs, feats
        else:
            return outputs


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            conv_layer_type: str = 'conv',
            norm_layer_type: str = 'none',
            activation_type: str = 'relu',
            resize_layer_type: str = 'none',
            efficient_upsampling: bool = False,  #
            return_feats: bool = False,
    ):
        """This is a base module for residual blocks"""
        super(ConvBlock, self).__init__()
        # Initialize layers in the block
        self.return_feats = return_feats
        m_bias = False

        if resize_layer_type in ['nearest', 'bilinear'] and stride > 1:
            self.upsample = lambda inputs: F.interpolate(inputs, scale_factor=stride, mode=resize_layer_type)

        downsample = resize_layer_type in downsampling_layers and stride > 1
        if downsample:
            downsampling_layer = downsampling_layers[resize_layer_type]

        normalize = norm_layer_type != 'none'
        if normalize:
            norm_layer = norm_layers[norm_layer_type]

        activation = activations[activation_type]
        conv_layer = conv_layers[conv_layer_type]

        if '3d' in conv_layer_type:
            num_kernel_dims = 3
        else:
            num_kernel_dims = 2

        ### Initialize the layers of the first half of the block ###
        layers = [
            conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size,) * num_kernel_dims,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=m_bias)]

        if normalize:
            layers += [norm_layer(out_channels)]

        layers += [activation(inplace=True)]

        self.block = nn.Sequential(*layers)

        if downsample:
            self.downsample = downsampling_layer(stride)

    def assign_spade_feats(self, feats):
        for m in self.modules():
            if m.__class__.__name__ == 'AdaptiveSPADE':
                m.feats = feats

    def forward(self, inputs, spade_feats=None):
        if spade_feats is not None:
            self.assign_spade_feats(spade_feats)

        if hasattr(self, 'upsample'):
            outputs = self.upsample(inputs)
        else:
            outputs = inputs

        feats = self.block(outputs)

        if hasattr(self, 'downsample'):
            outputs = self.downsample(feats)
        else:
            outputs = feats

        if self.return_feats:
            return outputs, feats
        else:
            return outputs





############################################################
#                Definitions for the layers                #
############################################################
class Conv2d_ws(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_ws, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv3d_ws(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv3d_ws, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        w = self.weight
        w_mean = w.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        w = w - w_mean
        std = w.view(w.size(0), -1).std(dim=1).view(-1,1,1,1,1) + 1e-5
        w = w / std.expand_as(w)
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)




# Supported blocks
blocks = {
    'res': ResBlock,
    'conv': ConvBlock
}

# Supported conv layers
conv_layers = {
    'conv': nn.Conv2d,
    # 'conv': Conv2d_ws,
    'conv_3d': nn.Conv3d,
    # 'conv_3d': Conv3d_ws,
    'ada_conv': AdaptiveConv,
    'ada_conv_3d': AdaptiveConv}

# Supported activations
activations = {
    'relu': nn.ReLU,
    # 'relu': functools.partial(nn.LeakyReLU, negative_slope=0.04),
    'lrelu': functools.partial(nn.LeakyReLU, negative_slope=0.2)}


# Supported normalization layers

norm_layers = {
    'in': lambda num_features, affine=True: nn.InstanceNorm2d(num_features=num_features, affine=affine),
    'bn': lambda num_features: nn.BatchNorm2d(num_features=num_features, momentum=MOMENTUM),
    'bn_3d': lambda num_features: nn.BatchNorm3d(num_features=num_features, momentum=MOMENTUM),
    'in_3d': lambda num_features, affine=True: nn.InstanceNorm3d(num_features=num_features, affine=affine),
    'sync_bn': lambda num_features: nn.SyncBatchNorm(num_features=num_features, momentum=MOMENTUM),
    'ada_in': lambda num_features, affine=True: AdaptiveInstanceNorm(num_features=num_features, affine=affine),
    'ada_bn': lambda num_features: AdaptiveBatchNorm(num_features=num_features, momentum=MOMENTUM),
    'ada_sync_bn': lambda num_features: AdaptiveSyncBatchNorm(num_features=num_features, momentum=MOMENTUM),
    'gn': lambda num_features, affine=True: nn.GroupNorm(num_groups=32, num_channels=num_features, affine=affine),
    'bcn': lambda num_features, affine=True: BCNorm(num_channels=num_features, num_groups=32, estimate=True),
    'bcn_3d': lambda num_features, affine=True: BCNorm(num_channels=num_features, num_groups=32,  estimate=True),
    'gn_24': lambda num_features, affine=True: nn.GroupNorm(num_groups=24, num_channels=num_features, affine=affine),
    'gn_3d': lambda num_features, affine=True: nn.GroupNorm(num_groups=32, num_channels=num_features, affine=affine),
    'ada_gn': lambda num_features, affine=True: AdaptiveGroupNorm(num_groups=32, num_features=num_features, affine=affine),
    # 'ada_gn': lambda num_features, affine=True: AdaptiveInstanceNorm(num_features=num_features, affine=affine),
    # 'ada_bcn': lambda num_features, affine=True: AdaptiveGroupNorm(num_groups=32, num_features=num_features, affine=affine),
    'ada_bcn': lambda num_features, affine=True: AdaptiveBCNorm(num_groups=32, num_features=num_features, estimate=True)
}
# Supported downsampling layers
downsampling_layers = {
    'avgpool': nn.AvgPool2d,
    'maxpool': nn.MaxPool2d,
    'avgpool_3d': nn.AvgPool3d,
    'maxpool_3d': nn.MaxPool3d,
    'pixelunshuffle': PixelUnShuffle}

class GridSample(nn.Module):
    def __init__(self, size):
        super(GridSample, self).__init__()
        self.size = size
        self.register_backward_hook(scale_warp_grad_norm)

    def forward(self, inputs, grid, padding_mode='reflection'):
        return F.grid_sample(inputs, grid, padding_mode=padding_mode)

def scale_warp_grad_norm(self, grad_input, grad_output):
    return grad_input[0], grad_input[1] / self.size

def assign_adaptive_norm_params(net_or_nets, params, alpha_norm=1.0):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if m_name in ['AdaptiveBatchNorm', 'AdaptiveSyncBatchNorm', 'AdaptiveInstanceNorm', 'AdaptiveGroupNorm', 'AdaptiveBCNorm']:  #TODO разобраться
            ada_weight, ada_bias = params.pop(0)

            m.ada_weight = m.weight[None] + ada_weight * alpha_norm
            m.ada_bias = m.bias[None] + ada_bias * alpha_norm

def replace_bn_to_in(module, name):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and bn to in
    for attr_str, _ in module.named_children():
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            # print('replaced: ', name, attr_str)
            new_bn = torch.nn.InstanceNorm2d(target_attr.num_features, target_attr.eps,
                                             target_attr.momentum, target_attr.affine,
                                          track_running_stats=False)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn_to_in(immediate_child_module, name)

    return module


def replace_bn_to_gn(module, name):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and bn to in
    # for attr_str in dir(module):
    for attr_str, _ in module.named_children():
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d or type(target_attr) == torch.nn.InstanceNorm2d:
            new_bn = torch.nn.GroupNorm(32, target_attr.num_features, target_attr.eps, target_attr.affine)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn_to_gn(immediate_child_module, name) #TODO поменять на GN

    return module

def replace_bn_to_bcn(module, name):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and bn to in
    # for attr_str in dir(module):
    for attr_str, _ in module.named_children():
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d or type(target_attr) == torch.nn.InstanceNorm2d:
            new_bn = BCNorm(32, target_attr.num_features, target_attr.eps)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn_to_bcn(immediate_child_module, name) #TODO поменять на GN

    return module


def replace_conv_to_ws_conv(module, conv2d=True, conv3d =True):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and bn to in
    # for attr_str in dir(module):
    prev_prev_attr = None
    prev_attr = None
    for indx, (attr_str, _) in enumerate(module.named_children()):

        if indx == 0:
            prev_prev_attr = getattr(module, attr_str)
        elif indx == 1:
            prev_attr = getattr(module, attr_str)
        else:
            # print(type(target_attr))
            target_attr = getattr(module, attr_str)
            if type(target_attr) == torch.nn.Conv2d and conv2d and (type(prev_prev_attr) == torch.nn.GroupNorm or type(prev_attr) == torch.nn.GroupNorm): #
                new_conv = Conv2d_ws(target_attr.in_channels, target_attr.out_channels, kernel_size=target_attr.kernel_size, stride = target_attr.stride, padding = target_attr.padding, dilation = target_attr.dilation,
                                     groups=target_attr.groups, bias=True)
                setattr(module, attr_str, new_conv)

            if type(target_attr) == torch.nn.Conv3d and conv3d and (type(prev_prev_attr) == AdaptiveGroupNorm or type(prev_attr) == AdaptiveGroupNorm): #
                new_conv = Conv3d_ws(target_attr.in_channels, target_attr.out_channels, kernel_size=target_attr.kernel_size, stride = target_attr.stride, padding = target_attr.padding, dilation = target_attr.dilation,
                                     groups=target_attr.groups, bias=True)
                setattr(module, attr_str, new_conv)
            prev_prev_attr = prev_attr
            prev_attr = target_attr

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_conv_to_ws_conv(immediate_child_module, name)

    return module

def apply_ws_to_nets(obj):
    ws_nets_names = args_utils.parse_str_to_list(obj.args.ws_networks, sep=',')
    for net_name in ws_nets_names:
        try:
            net = getattr(obj, net_name)
            new_net = replace_conv_to_ws_conv(net, conv2d=True, conv3d=True)
            setattr(obj, net_name, new_net)
            if obj.args.print_norms and obj.rank==0:
                print(f'WS applied to {net_name}')
        except Exception as e:
            pass
            # if obj.args.print_norms and obj.rank==0:
            #     print(e)


class ProjectorNorm(nn.Module):
    def __init__(self, net_or_nets,
                 eps,
                 gen_embed_size,
                 gen_max_channels):
        super(ProjectorNorm, self).__init__()
        self.eps = eps

        # Matrices that perform a lowrank matrix decomposition W = U E V
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            if m.__class__.__name__ in ['AdaptiveBatchNorm', 'AdaptiveSyncBatchNorm', 'AdaptiveInstanceNorm', 'AdaptiveGroupNorm', 'AdaptiveBCNorm'] :
                self.u += [nn.Parameter(torch.empty(m.num_features, gen_max_channels))]
                self.v += [nn.Parameter(torch.empty(gen_embed_size ** 2, 2))]

                nn.init.uniform_(self.u[-1], a=-math.sqrt(3 / gen_max_channels),
                                 b=math.sqrt(3 / gen_max_channels))
                nn.init.uniform_(self.v[-1], a=-math.sqrt(3 / gen_embed_size ** 2),
                                 b=math.sqrt(3 / gen_embed_size ** 2))

    def forward(self, embed_dict, iter=0):
        params = []

        for u, v in zip(self.u, self.v):
            embed = embed_dict['orig']

            param = u[None].matmul(embed).matmul(v[None])
            weight, bias = param.split(1, dim=2)

            params += [(weight[..., 0], bias[..., 0])]

        return params


class ProjectorNormLinear(nn.Module):
    def __init__(self, net_or_nets,
                 eps,
                 gen_embed_size,
                 gen_max_channels,
                 emb_v_exp=False,
                 no_detach_frec=1,
                 key_emb = 'orig'):
        super(ProjectorNormLinear, self).__init__()
        self.eps = eps
        self.emb_v_exp = emb_v_exp
        # Matrices that perform a lowrank matrix decomposition W = U E V
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()
        self.no_detach_frec = no_detach_frec

        self.key_emb = key_emb

        input_n = 512 if emb_v_exp else 512*16
        self.fc = nn.Sequential(
                    nn.Linear(input_n, 512, bias=False),
                    nn.ReLU(),
                    nn.Linear(512, 512*2, bias=False))
        
        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            if m.__class__.__name__ in ['AdaptiveBatchNorm', 'AdaptiveSyncBatchNorm', 'AdaptiveInstanceNorm', 'AdaptiveGroupNorm', 'AdaptiveBCNorm'] :
                self.u += [nn.Parameter(torch.empty(m.num_features, 512))]
                self.v += [nn.Parameter(torch.empty(2, 2))]

                nn.init.uniform_(self.u[-1], a=-math.sqrt(3 / 512),
                                 b=math.sqrt(3 / 512))
                nn.init.uniform_(self.v[-1], a=-math.sqrt(3 / 2 ),
                                 b=math.sqrt(3 / 2))

    def forward(self, embed_dict, iter=0):
        params = []
        if self.emb_v_exp:
            embed = embed_dict['ada_v'].detach() 
        else:
            embed = embed_dict[self.key_emb].view(-1, 512*16) if iter%self.no_detach_frec==0 else embed_dict[self.key_emb].view(-1, 512*16).detach()


        embed = self.fc(embed).view(-1, 512, 2)

        for u, v in zip(self.u, self.v):
            
            param = u[None].matmul(embed).matmul(v[None])
            weight, bias = param.split(1, dim=2)

            params += [(weight[..., 0], bias[..., 0])]

        return params


# class ProjectorNormLinear(nn.Module):
#     def __init__(self, net_or_nets,
#                  eps,
#                  gen_embed_size,
#                  gen_max_channels):
#         super(ProjectorNormLinear, self).__init__()
#         self.eps = eps

#         # Matrices that perform a lowrank matrix decomposition W = U E V
#         self.u = nn.ParameterList()
#         self.v = nn.ParameterList()

#         self.fc = nn.Sequential(
#                     nn.Linear(512, 512, bias=True),
#                     nn.ReLU(),
#                     nn.Linear(512, 512, bias=False))

#         # self.u_2 = nn.ParameterList()
#         # self.v_2 = nn.ParameterList()

#         if isinstance(net_or_nets, list):
#             modules = itertools.chain(*[net.modules() for net in net_or_nets])
#         else:
#             modules = net_or_nets.modules()

#         for m in modules:
#             if m.__class__.__name__ in ['AdaptiveBatchNorm', 'AdaptiveSyncBatchNorm', 'AdaptiveInstanceNorm', 'AdaptiveGroupNorm', 'AdaptiveBCNorm']:
#                 self.u += [nn.Parameter(torch.empty(m.num_features, gen_max_channels))]
#                 self.v += [nn.Parameter(torch.empty(gen_embed_size ** 2, 2))]

#                 nn.init.uniform_(self.u[-1], a=-math.sqrt(3 / gen_max_channels),
#                                  b=math.sqrt(3 / gen_max_channels))
#                 nn.init.uniform_(self.v[-1], a=-math.sqrt(3 / gen_embed_size ** 2),
#                                  b=math.sqrt(3 / gen_embed_size ** 2))

#         #         self.u_2 += [nn.Parameter(torch.empty(m.num_features, m.num_features))]
#         #         self.v_2 += [nn.Parameter(torch.empty(2, 2))]
#         #
#         #         nn.init.uniform_(self.u_2[-1], a=-math.sqrt(3 / gen_max_channels),
#         #                          b=math.sqrt(3 / gen_max_channels))
#         #         nn.init.uniform_(self.v_2[-1], a=-math.sqrt(3 / gen_embed_size ** 2),
#         #                          b=math.sqrt(3 / gen_embed_size ** 2))
#         # self.relu = nn.ReLU()
#     def forward(self, embed):
#         params = []

#         for u, v in zip(self.u, self.v):
#         # for u, v, u_2, v_2 in zip(self.u, self.v, self.u_2, self.v_2):
#             param = u[None].matmul(self.fc(*embed)).matmul(v[None])
#             # param = u_2[None].matmul(self.relu(u[None].matmul(embed).matmul(v[None]))).matmul(v_2[None])
#             weight, bias = param.split(1, dim=2)
#             params += [(weight[..., 0], bias[..., 0])]

#         return params



class ProjectorConv(nn.Module):
    def __init__(self, net_or_nets,
                 eps,
                 gen_adaptive_kernel,
                 gen_max_channels):
        super(ProjectorConv, self).__init__()
        self.eps = eps
        self.adaptive_kernel = gen_adaptive_kernel

        # Matrices that perform a lowrank matrix decomposition W = U E V
        self.u = nn.ParameterList()
        self.v = nn.ParameterList()
        self.kernel_size = []

        if isinstance(net_or_nets, list):
            modules = itertools.chain(*[net.modules() for net in net_or_nets])
        else:
            modules = net_or_nets.modules()

        for m in modules:
            if m.__class__.__name__ == 'AdaptiveConv':
                # Assumes that adaptive conv layers have no bias
                kernel_numel = m.kernel_size[0] * m.kernel_size[1]
                if len(m.kernel_size) == 3:
                    kernel_numel *= m.kernel_size[2]

                if kernel_numel == 1:
                    self.u += [nn.Parameter(torch.empty(m.out_channels, gen_max_channels // 2))]
                    self.v += [nn.Parameter(torch.empty(gen_max_channels // 2, m.in_channels))]

                elif kernel_numel > 1:
                    self.u += [nn.Parameter(torch.empty(m.out_channels, gen_max_channels // 2))]
                    self.v += [nn.Parameter(torch.empty(m.in_channels, gen_max_channels // 2))]

                self.kernel_size += [m.kernel_size]

                bound = math.sqrt(3 / (gen_max_channels // 2))
                nn.init.uniform_(self.u[-1], a=-bound, b=bound)
                nn.init.uniform_(self.v[-1], a=-bound, b=bound)

    def forward(self, embed_dict):
        params = []

        for u, v, kernel_size in zip(self.u, self.v, self.kernel_size):
            kernel_numel = kernel_size[0] * kernel_size[1]
            if len(kernel_size) == 3:
                kernel_numel *= kernel_size[2]

            if kernel_numel == 1:
                embed = embed_dict['fc']
            else:
                if self.adaptive_kernel:
                    if kernel_numel == 9:
                        embed = embed_dict['conv2d']
                    elif kernel_numel == 27:
                        embed = embed_dict['conv3d']
                    embed = embed.view(embed.shape[0], embed.shape[1], -1, kernel_numel)
                else:
                    embed = embed_dict['fc'][..., None]

            if kernel_numel == 1:
                # AdaptiveConv with kernel size = 1
                weight = u[None].matmul(embed).matmul(v[None])
                weight = weight.view(*weight.shape, *kernel_size)  # B x C_out x C_in x 1 ...
            else:
                # AdaptiveConv with kernel size > 1
                if self.adaptive_kernel:
                    kernel_numel_ = kernel_numel
                    kernel_size_ = kernel_size
                else:
                    kernel_numel_ = 1
                    kernel_size_ = (1,) * len(kernel_size)

                param = embed.view(*embed.shape[:2], -1)
                param = u[None].matmul(param)  # B x C_out x C_emb/2
                b, c_out = param.shape[:2]
                param = param.view(b, c_out, -1, kernel_numel_)
                param = v[None].matmul(param)  # B x C_out x C_in x kernel_numel
                weight = param.view(*param.shape[:3], *kernel_size_)

            params += [weight]

        return params


def assign_adaptive_conv_params(net_or_nets, params, adaptive_conv_type, alpha_conv=1.0):
    if isinstance(net_or_nets, list):
        modules = itertools.chain(*[net.modules() for net in net_or_nets])
    else:
        modules = net_or_nets.modules()

    for m in modules:
        m_name = m.__class__.__name__
        if m_name == 'AdaptiveConv':
            attr_name = 'weight_orig' if hasattr(m, 'weight_orig') else 'weight'
            weight = getattr(m, attr_name)
            ada_weight = params.pop(0)

            if adaptive_conv_type == 'sum':
                ada_weight = weight[None] + ada_weight * alpha_conv
            elif adaptive_conv_type == 'mul':
                ada_weight = weight[None] * (torch.sigmoid(ada_weight) * alpha_conv + (1 - alpha_conv))

            setattr(m, 'ada_' + attr_name, ada_weight)

class Face_vector(object):
    def __init__(self, head_pose_regressor, use_gpu=True, half=False):
        self.use_gpu = use_gpu
        self.head_pose_regressor = head_pose_regressor
        network = torchvision.models.vgg16(num_classes=2622).features
        state_dict = torch.utils.model_zoo.load_url(
            'http://www.robots.ox.ac.uk/~albanie/models/pytorch-mcn/'
            'vgg_face_dag.pth')
        layer_name_mapping = {
            0: 'conv1_1',
            2: 'conv1_2',
            5: 'conv2_1',
            7: 'conv2_2',
            10: 'conv3_1',
            12: 'conv3_2',
            14: 'conv3_3',
            17: 'conv4_1',
            19: 'conv4_2',
            21: 'conv4_3',
            24: 'conv5_1',
            26: 'conv5_2',
            28: 'conv5_3'}
        new_state_dict = {}
        for k, v in layer_name_mapping.items():
            new_state_dict[str(k) + '.weight'] = \
                state_dict[v + '.weight']
            new_state_dict[str(k) + '.bias'] = \
                state_dict[v + '.bias']
        network.load_state_dict(new_state_dict)
        self.network = network
        self.half = half
        if self.half:
            self.network.half()
        if self.use_gpu:
            self.network = network.cuda()

    def forward(self, image, crop=True, forward=True, S=0.5):
        if crop:
            image = F.interpolate(image, mode='bilinear', size=(256, 256), align_corners=False)
            grid_size = image.shape[2] // 2
            grid = torch.linspace(-1, 1, grid_size)
            v, u = torch.meshgrid(grid, grid)
            identity_grid = torch.stack([u, v, torch.ones_like(u)], dim=2).view(1, -1, 3)

            theta = self.head_pose_regressor.forward(image)[:, :3, :]
            eye_vector = torch.zeros(theta.shape[0], 1, 4)
            eye_vector[:, :, 3] = 1
            eye_vector = eye_vector.type(theta.type()).to(theta.device)
            theta_ = torch.cat([theta, eye_vector], dim=1)
            inv_theta_2d = theta_.inverse()[:, :, [0, 1, 3]][:, [0, 1, 3]]
            scale = torch.zeros_like(inv_theta_2d)
            scale[:, [0, 1], [0, 1]] = S
            scale[:, 2, 2] = 1
            inv_theta_2d = torch.bmm(inv_theta_2d, scale)[:, :2]
            n = image.shape[0]
            align_warp = identity_grid.repeat_interleave(n, dim=0)
            align_warp = align_warp.to(image.device)
            inv_theta_2d = inv_theta_2d.to(image.device)
            align_warp = align_warp.bmm(inv_theta_2d.transpose(1, 2)).view(n, grid_size, grid_size, 2)
            face_image = F.grid_sample(image.float(), align_warp.float())

            image = misc.apply_imagenet_normalization(face_image)
        if forward:
            if self.half:
                face_vector = self.network(image.half())
            else:
                face_vector = self.network(image)

            return face_vector, image
        else:
            return None, image



class Face_vector_resnet(object):
    # hyper parameters
    def __init__(self, use_gpu=True, half = False, project_dir='/fsx/nikitadrobyshev/latent-texture-avatar'):
        self.batch_size = 8
        self.half = half
        self.mean = (131.0912, 103.8827, 91.4953)
        self.use_gpu = use_gpu
        self.project_dir = project_dir
        self.model_eval = self.initialize_model()
        


    def chunks(self, l, n):
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            yield l[i:i + n]

    def initialize_model(self):
        # Download the pytorch model and weights.
        # Currently, it's cpu mode.
        from .senet50_ft_dag import senet50_ft_dag
        network = senet50_ft_dag(weights_path=f'{self.project_dir}/losses/loss_model_weights/senet50_ft_dag.pth')
        if self.half:
            network.half()
        network.eval()
        if self.use_gpu:
            network = network.cuda()
        return network


    def image_encoding(self, model, img):

        num_faces = img.shape[0]

        if self.half:
            model.half()

        imgchunks = list(self.chunks(img, self.batch_size))
        # face_feats = torch.zeros((num_faces, 2048))
        for c, imgs in enumerate(imgchunks):
            if self.half:
                f = model(imgs.half())[1][:, :, 0, 0]
            else:
                f = model(imgs)[1][:, :, 0, 0]
            f = f / torch.sqrt(torch.sum(f ** 2))
        return f

    def forward(self, img):
        # print(img.max(), img.min())
        img = img * 255
        inputs = F.interpolate(img, mode='bilinear', size=(244, 244), align_corners=False)
        mean = inputs.new_tensor(self.mean).view(1, 3, 1, 1)
        inputs = inputs - mean
        face_feats = self.image_encoding(self.model_eval, inputs)
        return face_feats


#######################################################################################################################
# StyleGAN2 stuff

from ..basic_avatar.op import conv2d_gradfix

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss






def _calc_r1_penalty(real_images: torch.Tensor,
                     scores_real: List[torch.Tensor],
                     scale_number: Union[int, List[int], str] = 0,
                     ) -> torch.Tensor:
    assert real_images.requires_grad
    if isinstance(scale_number, int):
        scale_number = [scale_number]
    if isinstance(scale_number, str):
        if scale_number != 'all':
            raise ValueError(f'scale_number should be int, List[int] or literal "all". Got value: {scale_number}')
        scale_number = list(range(len(scores_real)))

    penalties = 0.
    for scale_idx in scale_number:
        scores = scores_real[scale_idx]
        gradients = torch.autograd.grad(scores.sum(), real_images, create_graph=True, retain_graph=True)[0]
        penalty = gradients.pow(2).view(gradients.shape[0], -1).sum(1).mean()
        penalties += penalty
    return penalties / len(scale_number)