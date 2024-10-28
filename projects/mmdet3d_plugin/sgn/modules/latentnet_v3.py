import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from torch.distributions import Normal, Independent, kl
from .latent_v3 import Decoder, Encoder_xy, Encoder_x

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF


class LatentNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Hyperparemeters     
        channel = 32
        latent_dim = 3
        
        
        self.x_encoder = Encoder_x(4, channel, latent_dim)
        self.xy_encoder = Encoder_xy(4, channel, latent_dim)
        self.decoder = Decoder(latent_dim)
        
    
    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)
    
    def forward_train(self, img, img_metas, target):
        '''
        input:
            img: torch.Size([1, 5, 3, 370, 1220])
            depth: torch.Size([5, 1, 370, 1220])
        '''
        device = target.device
        
        img = img[0].to(device)  # torch.Size([5, 3, 370, 1220])
        depth = img_metas[0]['depth_tensor'].to(device)  # torch.Size([5, 1, 370, 1220])
        
        combined_image = torch.cat((img, depth),1)
        # Encoder x
        self.prior, mux, logvarx = self.x_encoder(combined_image)  # input: torch.Size([5, 4, 370, 1220])
        
        #z_noise_prior = self.reparametrize(mux, logvarx)  # torch.Size([5, 3])
        
        
        # Encoder xy
        self.posterior, muxy, logvarxy = self.xy_encoder(combined_image, target)
        
        z_noise_post = self.reparametrize(muxy, logvarxy) # torch.Size([5, 3])
        
        img_metas[0]['z'] = z_noise_post
        
        img = self.decoder.forward_image(img, depth, z_noise_post)
        
        loss_dict = dict()
        lattent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))
        loss_dict['loss_latent'] = lattent_loss
        
        return loss_dict, img
    
    def forward_test(self, img, img_metas, target):
        '''
        input:
            img: torch.Size([1, 5, 3, 370, 1220])
            depth: torch.Size([5, 1, 370, 1220])
        '''
        device = target.device
        
        img = img[0].to(device)  # torch.Size([5, 3, 370, 1220])
        depth = img_metas[0]['depth_tensor'].to(device)  # torch.Size([5, 1, 370, 1220])
        
        # Encoder x
        self.prior, mux, logvarx = self.x_encoder(torch.cat((img, depth),1))  # input: torch.Size([5, 4, 370, 1220])
        
        z_noise_prior = self.reparametrize(mux, logvarx)  # torch.Size([5, 3])
        
        img_metas[0]['z'] = z_noise_prior
        
        img = self.decoder.forward_image(img, depth, z_noise_prior)
        
        return img
        
