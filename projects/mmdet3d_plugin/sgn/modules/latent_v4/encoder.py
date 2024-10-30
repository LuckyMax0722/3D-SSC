import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Normal, Independent, kl
from .voxel_head import UNet

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF

class Encoder_x(nn.Module):
    def __init__(self, latent_size):
        super(Encoder_x, self).__init__()
        
        # 2 Voxel Branch
        voxel_channel = 128
        
        self.voxel_encoder1 = nn.Sequential(
            nn.Conv3d(int(voxel_channel), int(voxel_channel*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*1.5), int(voxel_channel*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*1.5), int(voxel_channel*2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*2), int(voxel_channel*2), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*2), int(voxel_channel*2.5), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*2.5), int(voxel_channel*2.5), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*2.5), int(voxel_channel*3), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*3), int(voxel_channel*3), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 3. Combine Branch
        self.fc1 = nn.Linear(384 * 16 * 16 * 2, latent_size)
        self.fc2 = nn.Linear(384 * 16 * 16 * 2, latent_size)
        
    def forward(self, x3d_input):
        '''
        input:
            x3d: torch.Size([1, 128, 128, 128, 16])
        '''
         # 1.Voxel Branch
        
        x3d_input = self.voxel_encoder1(x3d_input) # torch.Size([1, 192, 128, 128, 16])
        x3d_input = self.voxel_encoder2(x3d_input) # torch.Size([1, 256, 64, 64, 8])
        x3d_input = self.voxel_encoder3(x3d_input) # torch.Size([1, 320, 32, 32, 4])
        x3d_input = self.voxel_encoder4(x3d_input) # torch.Size([1, 384, 16, 16, 2])
        
        x3d_input = x3d_input.view(1, 384 * 16 * 16 * 2)  # [1, 196,608]
        
        mu = self.fc1(x3d_input)
        logvar = self.fc2(x3d_input)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return dist, mu, logvar

class Encoder_xy_spconv(nn.Module):
    def __init__(self, latent_size):
        super(Encoder_xy, self).__init__()
        
        # 1. Hyperparemeters
        
        # 2 Voxel Branch
        voxel_channel = 128
        
        self.voxel_encoder1 = nn.Sequential(
            nn.Conv3d(int(voxel_channel), int(voxel_channel*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*1.5), int(voxel_channel*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*1.5), int(voxel_channel*2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*2), int(voxel_channel*2), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*2), int(voxel_channel*2.5), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*2.5), int(voxel_channel*2.5), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*2.5), int(voxel_channel*3), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*3), int(voxel_channel*3), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # backbone3d
        num_point_features = 1
        gird_size = CONF.KITTI.GRID_SIZE
        
        gird_size = np.array(gird_size)
        self.backbone3d = UNet(num_point_features, gird_size)
        
        # 3. Combine Branch
        self.fc1 = nn.Linear(384 * 16 * 16 * 2 + 128 * 32 *32, latent_size)
        self.fc2 = nn.Linear(384 * 16 * 16 * 2 + 128 * 32 *32, latent_size)
        
        
    def forward(self, x3d_input, target_input):
        '''
        input:
            target: torch.Size([1, 256, 256, 32])
            x3d: torch.Size([1, 128, 262144])
        '''
        
        device = target_input.device
        
        # 1.Voxel Branch
        
        x3d_input = self.voxel_encoder1(x3d_input) # torch.Size([1, 192, 128, 128, 16])
        x3d_input = self.voxel_encoder2(x3d_input) # torch.Size([1, 256, 64, 64, 8])
        x3d_input = self.voxel_encoder3(x3d_input) # torch.Size([1, 320, 32, 32, 4])
        x3d_input = self.voxel_encoder4(x3d_input) # torch.Size([1, 384, 16, 16, 2])
        
        x3d_input = x3d_input.view(1, 384 * 16 * 16 * 2)  # [1, 196,608]
        
        # 2.Target Branch
        
        target_input = target_input[0].permute(2, 1, 0)  # [x, y, z] --> [z, y, x]  [32, 256, 256]

        mask = (target_input >= 0) & (target_input <= 19)

        coords = torch.nonzero(mask, as_tuple=False)  # 0-19 voxel coord  torch tensor [N, 3]
        values = target_input[mask].unsqueeze(1)  # class feature torch tensor [N, 1]
        
        zeros_column = torch.zeros((coords.shape[0], 1), dtype=coords.dtype).to(device)  # bs
        coords = torch.cat([zeros_column, coords], dim=1)  # torch tensor [N, 4]
        
        spatial_features = self.backbone3d(values, coords)  # torch.Size([1, 128, 1, 32, 32])
        
        spatial_features = spatial_features.view(1, 128 * 32 * 32)  # [1, 131,072]

        # 3.Combined Branch
        combined_features = torch.cat((x3d_input, spatial_features), dim=1)  # torch.Size([1, 196,608 + 131,072])
        
        mu = self.fc1(combined_features)
        logvar = self.fc2(combined_features)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return dist, mu, logvar
    
class Encoder_xy(nn.Module):
    def __init__(self, latent_size):
        super(Encoder_xy, self).__init__()
        
        # 1. Hyperparemeters
        voxel_channel = 128
        target_channel = 1
        # 2 Voxel Branch
        
        
        self.voxel_encoder1 = nn.Sequential(
            nn.Conv3d(int(voxel_channel), int(voxel_channel*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*1.5), int(voxel_channel*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*1.5), int(voxel_channel*2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*2), int(voxel_channel*2), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*2), int(voxel_channel*2.5), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*2.5), int(voxel_channel*2.5), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.voxel_encoder4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(voxel_channel*2.5), int(voxel_channel*3), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel*3), int(voxel_channel*3), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 2.5 Target Branch
        self.target_encoder1 = nn.Sequential(
            nn.Conv3d(int(target_channel), int(target_channel*2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(target_channel*2), int(target_channel*2), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.target_encoder2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(target_channel*2), int(target_channel*4), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(target_channel*4), int(target_channel*4), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.target_encoder3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(target_channel*4), int(target_channel*8), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(target_channel*8), int(target_channel*8), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.target_encoder4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(int(target_channel*8), int(target_channel*16), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(target_channel*16), int(target_channel*16), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        
        # 3. Combine Branch
        self.fc1 = nn.Linear(384 * 16 * 16 * 2 + 16 * 4 * 32 * 32, latent_size)
        self.fc2 = nn.Linear(384 * 16 * 16 * 2 + 16 * 4 * 32 * 32, latent_size)
        
        
    def forward(self, x3d_input, target_input):
        '''
        input:
            target: torch.Size([1, 256, 256, 32])
            x3d: torch.Size([1, 128, 128, 128, 16])
        '''
        
        # 1.Voxel Branch
        
        x3d_input = self.voxel_encoder1(x3d_input) # torch.Size([1, 192, 16, 128, 128])
        x3d_input = self.voxel_encoder2(x3d_input) # torch.Size([1, 256, 8, 64, 64])
        x3d_input = self.voxel_encoder3(x3d_input) # torch.Size([1, 320, 4, 32, 32])
        x3d_input = self.voxel_encoder4(x3d_input) # torch.Size([1, 384, 2, 16, 16])
        
        x3d_input = x3d_input.view(1, 384 * 16 * 16 * 2)  # [1, 196,608]
        
        # 2.Target Branch
        
        target_input = self.target_encoder1(target_input)  # torch.Size([1, 2, 32, 256, 256])
        target_input = self.target_encoder2(target_input)  # torch.Size([1, 4, 16, 128, 128])
        target_input = self.target_encoder3(target_input)  # torch.Size([1, 8, 8, 64, 64])
        target_input = self.target_encoder4(target_input)  # torch.Size([1, 16, 4, 32, 32])

        target_input = target_input.view(1, 16 * 4 * 32 *32)  # [1, 16,384]
        
        # 3.Combined Branch
        combined_features = torch.cat((x3d_input, target_input), dim=1)  # torch.Size([1, 196,608 + 16,384])
        
        mu = self.fc1(combined_features)
        logvar = self.fc2(combined_features)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return dist, mu, logvar