import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Normal, Independent, kl
from .voxel_head import VoxelBackBone8x, HeightCompression, BaseBEVBackbone

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF

class Encoder_x(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 11 * 38, latent_size)
        self.fc2 = nn.Linear(channels * 8 * 11 * 38, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))  # torch.Size([5, 32, 185, 610])
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))  # torch.Size([5, 64, 92, 305])
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))  # torch.Size([5, 128, 46, 152])
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))  # torch.Size([5, 256, 23, 76])
        # print(output.size())
        output = self.leakyrelu(self.bn5(self.layer5(output)))  # torch.Size([5, 256, 11, 38])
        # print(output.size())
        output = output.view(-1, self.channel * 8 * 11 * 38)  # torch.Size([5, 107008])
        # print(output.size())
        # output = self.tanh(output)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return dist, mu, logvar

class Encoder_xy(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_xy, self).__init__()
        
        # 1. Hyperparemeters
        
        # 2 Voxel Branch
        # backbone3d
        num_point_features = 1
        gird_size = CONF.KITTI.GRID_SIZE
        
        # backbone2d
        bev_input_channels = CONF.PVRCNN.NUM_BEV_FEATURES 
        layer_nums = CONF.PVRCNN.LAYER_NUMS
        layer_strides = CONF.PVRCNN.LAYER_STRIDES
        num_filters = CONF.PVRCNN.NUM_FILTERS
        upsample_strides = CONF.PVRCNN.UPSAMPLE_STRIDES
        num_upsample_filters = CONF.PVRCNN.NUM_UPSAMPLE_FILTERS
        
        gird_size = np.array(gird_size)
        self.backbone3d = VoxelBackBone8x(num_point_features, gird_size)
        self.view_transform = HeightCompression()
        self.backbone2d = BaseBEVBackbone(bev_input_channels, layer_nums, layer_strides, num_filters, upsample_strides, num_upsample_filters)
        
        
        # Image Branch
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        
        # Combined Branch
        self.fc1 = nn.Linear(channels * 8 * 11 * 38 + 128 * 32 * 32, latent_size)
        self.fc2 = nn.Linear(channels * 8 * 11 * 38 + 128 * 32 * 32, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self, image_input, target_input):
        '''
        input:
            target: torch.Size([1, 256, 256, 32])
        '''
        
        # 1.Voxel Branch
        device = target_input.device
        target_input = target_input[0].permute(2, 1, 0)  # [x, y, z] --> [z, y, x]  [32, 256, 256]

        mask = (target_input >= 0) & (target_input <= 19)

        coords = torch.nonzero(mask, as_tuple=False)  # 0-19 voxel coord  torch tensor [N, 3]
        values = target_input[mask].unsqueeze(1)  # class feature torch tensor [N, 1]
        
        zeros_column = torch.zeros((coords.shape[0], 1), dtype=coords.dtype).to(device)  # bs
        coords = torch.cat([zeros_column, coords], dim=1)  # torch tensor [N, 4]
        
        spatial_features = self.backbone3d(values, coords)  # [1024, 128]
        spatial_features = self.view_transform(spatial_features)  # torch.Size([1, 128, 32, 32])
        spatial_features = self.backbone2d(spatial_features) # torch.Size([1, 128, 32, 32])
        
        spatial_features = spatial_features.view(-1, 128 * 32 * 32)  # torch.Size([1, 131072])
        spatial_features = spatial_features.expand(5, 128 * 32 * 32)  # torch.Size([5, 131072])

        # 2. Image Branch
        output = self.leakyrelu(self.bn1(self.layer1(image_input)))  # torch.Size([5, 32, 185, 610])
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))  # torch.Size([5, 64, 92, 305])
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))  # torch.Size([5, 128, 46, 152])
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))  # torch.Size([5, 256, 23, 76])
        # print(output.size())
        output = self.leakyrelu(self.bn5(self.layer5(output)))  # torch.Size([5, 256, 11, 38])
        # print(output.size())
        output = output.view(-1, self.channel * 8 * 11 * 38)  # torch.Size([5, 107008])
        # print(output.size())
        
        # 3.Combined Branch
        output = torch.cat((output, spatial_features), dim=1)  # torch.Size([5, 107008 + 131072])
        
        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        
        return dist, mu, logvar