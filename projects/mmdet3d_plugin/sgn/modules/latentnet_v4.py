import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from torch.distributions import Normal, Independent, kl
from .latent_v4 import Decoder, Encoder_xy, Encoder_x

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF


class LatentNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Hyperparemeters     
        latent_dim = 128
        
        self.x_encoder = Encoder_x(latent_dim)
        self.xy_encoder = Encoder_xy(latent_dim)
        
        self.decoder = Decoder(latent_dim)

    
    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)
    
    def forward_train(self, x3d, target):
        '''
        input:
            target: torch.Size([1, 256, 256, 32])
            x3d: torch.Size([1, 128, 128, 128, 16])
        '''
        
        # dim change
        target = target.unsqueeze(1).permute(0, 1, 4, 3, 2)  # torch.Size([1, 1, 32, 256, 256])
        x3d = x3d.permute(0, 1, 4, 3, 2)  # torch.Size([1, 128, 16, 128, 128]      
        
        '''
            target: torch.Size([1, 1, 32, 256, 256])
            x3d: torch.Size([1, 128, 16, 128, 128])
        '''
        
        # Encoder x
        self.prior, mux, logvarx = self.x_encoder(x3d)
        
        #z_noise_prior = self.reparametrize(mux, logvarx)
        
        # Encoder xy
        self.posterior, muxy, logvarxy = self.xy_encoder(x3d, target)
        
        z_noise_post = self.reparametrize(muxy, logvarxy) 
        
        x3d = self.decoder.forward_voxel(x3d, z_noise_post)
        
        lattent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))
        
        return x3d, lattent_loss

    
    def forward_test(self, x3d):
        '''
        input:
            x3d: torch.Size([1, 128, 128, 128, 16])
        '''
        
        x3d = x3d.permute(0, 1, 4, 3, 2)  # torch.Size([1, 128, 16, 128, 128]      
        
        '''
            x3d: torch.Size([1, 128, 16, 128, 128])
        '''
        
        # Encoder x
        self.prior, mux, logvarx = self.x_encoder(x3d)
        
        z_noise_prior = self.reparametrize(mux, logvarx)
        
        x3d = self.decoder.forward_voxel(x3d, z_noise_prior)
        
        return x3d
        
