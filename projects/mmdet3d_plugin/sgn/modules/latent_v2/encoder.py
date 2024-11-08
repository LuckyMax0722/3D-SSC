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
        
        self.channels = channels
        
        self.image_encoderx_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderx_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderx_3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderx_4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderx_5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 8, channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderx_6 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 8, channels * 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 16, channels * 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.fc1 = nn.Linear(channels * 16 * 11 * 38, latent_size)
        self.fc2 = nn.Linear(channels * 16 * 11 * 38, latent_size)


    def forward(self, input):
        output = self.image_encoderx_1(input)  # torch.Size([5, 3, 370, 1220])
        #print(output.size())
        output = self.image_encoderx_2(output)  # torch.Size([5, 8, 185, 610])
        #print(output.size())
        output = self.image_encoderx_3(output)  # torch.Size([5, 16, 92, 305])
        #print(output.size())
        output = self.image_encoderx_4(output)  # torch.Size([5, 32, 46, 152])
        #print(output.size())
        output = self.image_encoderx_5(output)  # torch.Size([5, 64, 23, 76])
        #print(output.size())
        output = self.image_encoderx_6(output)  # torch.Size([5, 128, 11, 38])
        #print(output.size())
        
        output = output.view(-1, self.channels * 16 * 11 * 38)  # torch.Size([5, 53504])
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
        self.channel = channels
        
        self.image_encoderxy_1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderxy_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderxy_3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderxy_4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderxy_5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 8, channels * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        
        self.image_encoderxy_6 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channels * 8, channels * 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels * 16, channels * 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        
        # Combined Branch
        self.fc1 = nn.Linear(channels * 16 * 11 * 38 + 128 * 32 * 32, latent_size)
        self.fc2 = nn.Linear(channels * 16 * 11 * 38 + 128 * 32 * 32, latent_size)
        
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
        output = self.image_encoderxy_1(image_input)  # torch.Size([5, 3, 370, 1220])
        #print(output.size())
        output = self.image_encoderxy_2(output)  # torch.Size([5, 8, 185, 610])
        #print(output.size())
        output = self.image_encoderxy_3(output)  # torch.Size([5, 16, 92, 305])
        #print(output.size())
        output = self.image_encoderxy_4(output)  # torch.Size([5, 32, 46, 152])
        #print(output.size())
        output = self.image_encoderxy_5(output)  # torch.Size([5, 64, 23, 76])
        #print(output.size())
        output = self.image_encoderxy_6(output)  # torch.Size([5, 128, 11, 38])
        #print(output.size())
        
        output = output.view(-1, self.channel * 16 * 11 * 38)
        
        # 3.Combined Branch
        output = torch.cat((output, spatial_features), dim=1)  # torch.Size([5, 107008 + 131072])
        
        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        
        return dist, mu, logvar