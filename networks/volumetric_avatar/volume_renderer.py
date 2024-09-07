import torch
from torch import nn
import numpy as np
from .volume_render_utils import ImportanceRenderer
from dataclasses import dataclass

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, out_features, pos_encoding = False, multires=10, features_sigm=1, squeeze_dim=0, depth_resolution=48, hidden_dim=448, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.squeeze_dim = squeeze_dim
        self.pos_encoding = pos_encoding
        out_dim = 3
        if self.pos_encoding:
            self.embedder, out_dim = get_embedder(multires=multires)
        self.features_sigm = features_sigm
        
        coord_dim = out_dim if pos_encoding else 3
        self.depth_resolution = depth_resolution

        input_d = n_features
        if self.squeeze_dim>0:
            self.squeeze = nn.Conv2d(
                    in_channels=n_features,
                    out_channels=self.squeeze_dim,
                    kernel_size=(1, 1),
                    bias=False)
            input_d = self.squeeze_dim


        model_list = [nn.Linear(input_d+coord_dim, self.hidden_dim),
            torch.nn.Softplus(),
        ]

        for i in range(num_layers-2):
            model_list+=[nn.Linear(hidden_dim, self.hidden_dim),
                          torch.nn.Softplus(),
                          ]

        model_list.append(nn.Linear(self.hidden_dim, 1 + out_features))



        self.net = torch.nn.Sequential(*model_list)


    def forward(self, coordinates, sampled_features):
        # Aggregate features
        # print(coordinates.shape, '111')
        if self.pos_encoding:
            coordinates = self.embedder(coordinates)

        CC = 16 * 96
        N, HW, CD, C = coordinates.shape



        coordinates = coordinates.reshape(N, 64, 64, CD, C)
        sampled_features = sampled_features.reshape(N, CC, 64, 64)

        if self.squeeze_dim > 0:
            CC = 512
            sampled_features = self.squeeze(sampled_features)
        sampled_features = sampled_features.permute(0, 2, 3, 1)

        X = []
        for i in range(64):
            # for j in range(64):
            # print(sampled_features[:, i].shape)
            b = sampled_features[:, i].shape[0]
            cur_f = sampled_features[:, i].reshape(b*64, 1, CC).repeat(1, CD, 1)
            cur_cor = coordinates[:, i].reshape(b*64, CD, 3)
            x = torch.cat([cur_f, cur_cor], dim=-1)
            N, M, C = x.shape
            # print(N, M, C)

            x = x.view(N * M, C)

            x = self.net(x)
            x = x.view(b, 64*self.depth_resolution, -1)
            X.append(x)
        X = torch.cat(X, dim=1).view(b, 64*64*self.depth_resolution, -1)
        # print(X.shape)
        if self.features_sigm:
            rgb = torch.sigmoid(X[..., 1:]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        else:
            # print('no sigm')
            rgb = torch.sigmoid(X[..., 1:4]) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
            rgb = torch.cat([rgb, X[..., 4:]], dim=-1)

        sigma = X[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}


class VolumeRenderer(nn.Module):
    
    @dataclass
    class Config:
        z_dim: int = 16,  # Input latent (Z) dimensionality.
        c_dim: int = 96,  # Conditioning label (C) dimensionality.
        w_dim: int = 64,  # Intermediate latent (W) dimensionality.
        img_resolution: int = 64,  # Output resolution.
        dec_channels: int = 1024,  # Number of output color channels
        img_channels: int = 384,  # Number of output color channels
        features_sigm: int  = 1,
        squeeze_dim: int = 0,
        depth_resolution: int = 48,
        hidden_vol_dec_dim: int = 448,
        num_layers_vol_dec: int  = 2

    def __init__(self, cfg):
        super(VolumeRenderer, self).__init__()
        self.cfg = cfg
        self.z_dim=self.cfg.z_dim
        self.c_dim=self.cfg.c_dim
        self.w_dim=self.cfg.w_dim
        self.img_resolution=self.cfg.img_resolution
        self.depth_resolution = self.cfg.depth_resolution
        self.img_channels=self.cfg.img_channels
        self.decoder = OSGDecoder(self.cfg.dec_channels, self.cfg.img_channels, features_sigm=self.cfg.features_sigm, hidden_dim=self.cfg.hidden_vol_dec_dim,
                                  squeeze_dim=self.cfg.squeeze_dim, depth_resolution=self.cfg.depth_resolution, num_layers=self.cfg.num_layers_vol_dec)
        self.renderer = ImportanceRenderer(depth_resolution=self.cfg.depth_resolution)

    def forward(self, aligned_target_volume):
        N, _, _, H, W = aligned_target_volume.shape
        feature_samples, depth_samples, weights_samples = self.renderer(aligned_target_volume, self.decoder)  # channels last
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        rgb_image = feature_image[:, :3]

        return feature_image, rgb_image, depth_image


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires=10, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim