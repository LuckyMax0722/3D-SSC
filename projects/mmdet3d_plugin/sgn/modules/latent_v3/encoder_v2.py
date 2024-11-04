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
        voxel_channel = 1
        
        self.voxel_encoder1 = nn.Sequential(
            nn.Conv3d(int(voxel_channel), int(voxel_channel), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel), int(voxel_channel), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*1), int(voxel_channel*4), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*4), int(voxel_channel*4), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*4), int(voxel_channel*16), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*16), int(voxel_channel*16), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*16), int(voxel_channel*64), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*64), int(voxel_channel*64), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        
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
        self.fc1 = nn.Linear(channels * 8 * 11 * 38 + 64 * 4 * 32 * 32, latent_size)
        self.fc2 = nn.Linear(channels * 8 * 11 * 38 + 64 * 4 * 32 * 32, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self, image_input, target_input):
        '''
        input:
            target: torch.Size([256, 256, 32])
        '''
        
        # 1.Voxel Branch
        target_input = target_input[0].permute(2, 1, 0)  # [x, y, z] --> [z, y, x]  [32, 256, 256]

        target_input = target_input.unsqueeze(0).unsqueeze(1)  # torch.Size([1, 1, 32, 256, 256])
        #print(target_input.size())
        
        target_input = self.voxel_encoder1(target_input)  # torch.Size([1, 1, 32, 256, 256])
        # print(target_input.size())
        target_input = self.voxel_encoder2(target_input)  # torch.Size([1, 4, 16, 128, 128])
        # print(target_input.size())
        target_input = self.voxel_encoder3(target_input)  # torch.Size([1, 16, 8, 64, 64])
        # print(target_input.size())
        target_input = self.voxel_encoder4(target_input)  # torch.Size([1, 64, 4, 32, 32])
        # print(target_input.size())
        target_input = target_input.view(-1, 64 * 4 * 32 * 32)  # torch.Size([1, 262144])
        # print(target_input.size())
        target_input = target_input.expand(5, 64 * 4 * 32 * 32) # torch.Size([5, 262144])
        # print(target_input.size())

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
        output = torch.cat((output, target_input), dim=1)  # torch.Size([5, 107008 + 131072])
        
        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        
        return dist, mu, logvar