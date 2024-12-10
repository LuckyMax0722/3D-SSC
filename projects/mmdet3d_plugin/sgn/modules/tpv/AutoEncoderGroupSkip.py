import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import HEADS

class Encoder(nn.Module):
    def __init__(self, geo_feat_channels, z_down, padding_mode, kernel_size = (5, 5, 3), padding = (2, 2, 1)):
        super().__init__()
        self.z_down = z_down
        self.conv0 = nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode)
        self.convblock1 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )
        if self.z_down :
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        else :
            self.downsample = nn.Sequential(
                nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0), bias=True, padding_mode=padding_mode),
                nn.InstanceNorm3d(geo_feat_channels)
            )
        self.convblock2 = nn.Sequential(
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels),
            nn.LeakyReLU(1e-1, True),
            nn.Conv3d(geo_feat_channels, geo_feat_channels, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding, bias=True, padding_mode=padding_mode),
            nn.InstanceNorm3d(geo_feat_channels)
        )

    def forward(self, x):  # [b, geo_feat_channels, X, Y, Z]
        x = self.conv0(x)  # [b, geo_feat_channels, X, Y, Z]

        residual_feat = x
        x = self.convblock1(x)  # [b, geo_feat_channels, X, Y, Z]
        x = x + residual_feat   # [b, geo_feat_channels, X, Y, Z]
        x = self.downsample(x)  # [b, geo_feat_channels, X//2, Y//2, Z//2]

        residual_feat = x
        x = self.convblock2(x)
        x = x + residual_feat

        return x  # [b, geo_feat_channels, X//2, Y//2, Z//2]

@HEADS.register_module()
class AutoEncoderGroupSkipV2(BaseModule):
    def __init__(
        self, 
        num_class,
        geo_feat_channels,
        z_down,
        padding_mode,
        voxel_fea,
        pos,
        feat_channel_up,
        triplane,
        mlp_hidden_channels,
        mlp_hidden_layers,
        dataset,
    ):
        
        super().__init__()
        
        class_num = num_class 
        self.embedding = nn.Embedding(class_num, geo_feat_channels)

        if dataset == 'kitti':
            self.geo_encoder = Encoder(geo_feat_channels, z_down, padding_mode)
        else:
            self.geo_encoder = Encoder(geo_feat_channels, z_down, padding_mode, kernel_size = 3, padding = 1)

        if voxel_fea :
            self.norm = nn.InstanceNorm3d(geo_feat_channels) 
        else:
            self.norm = nn.InstanceNorm2d(geo_feat_channels)
            
        self.geo_feat_dim = geo_feat_channels
        self.pos = pos
        self.pos_num_freq = 6  # the defualt value 6 like NeRF
        self.voxel_fea = voxel_fea
        self.triplane = triplane

    def encode(self, vol):
        x = vol.detach().clone()
        x[x == 255] = 0
            
        x = self.embedding(x)
        x = x.permute(0, 4, 1, 2, 3)
        vol_feat = self.geo_encoder(x)

        if self.voxel_fea:
            vol_feat = self.norm(vol_feat).tanh()
            return vol_feat
        else :
            xy_feat = vol_feat.mean(dim=4)
            xz_feat = vol_feat.mean(dim=3)
            yz_feat = vol_feat.mean(dim=2)
            
            xy_feat = (self.norm(xy_feat) * 0.5).tanh()
            xz_feat = (self.norm(xz_feat) * 0.5).tanh()
            yz_feat = (self.norm(yz_feat) * 0.5).tanh()
            return [xy_feat, xz_feat, yz_feat]

    def forward(self, vol):
        feat_map = self.encode(vol)
        return feat_map

    