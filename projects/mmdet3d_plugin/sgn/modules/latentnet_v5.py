import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from torch.distributions import Normal, Independent, kl
from .latent_v5 import vqvae, mlp_unet

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF


class LatentNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Hyperparemeters     
        self.encoder_y = vqvae(
            init_size=CONF.SSD.init_size, 
            num_classes=CONF.SSD.num_classes, 
            vq_size=CONF.SSD.vq_size, 
            l_size=CONF.SSD.l_size, 
            l_attention=CONF.SSD.l_attention
        )
        
        self.encoder_x = mlp_unet(
            channel=CONF.SGN.embed_dims, # 128 
            out_channel=CONF.SGN.embed_dims // 2, # 64 
            feature=CONF.SGN.embed_dims // 2,  # 64
            class_num=CONF.SGN.class_num, # 20
        )

        self.class_num = CONF.SGN.class_num # 20
        self.l_size = CONF.SSD.l_size
        self.fc1 = nn.Linear(CONF.SSD.out_dim, CONF.SSD.out_dim)
        self.fc2 = nn.Linear(CONF.SSD.out_dim, CONF.SSD.out_dim)
        
    
    def dist(self, x):
        mu = self.fc1(x)
        logvar = self.fc2(x)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        
        return mu, logvar, dist
            
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
            target_2: torch.Size([1, 128, 128, 16])
            x3d: torch.Size([1, 128, 128, 128, 16])
        '''    
        
        # 1. Target Branch

        target_20 = torch.where(target == 255, 0, target.long())
        
        latent_post, vq_loss_post, recons_loss_post = self.encoder_y.forward_post(target_20)  # --> torch.Size([1, 21, 32, 32, 4])

        if self.l_size == '32322':
            latent_post = latent_post.view(1, 20, 32 * 32 * 8)
        elif self.l_size== '16162':
            latent_post = latent_post.view(1, 20, 16 * 16 * 4)
        elif self.l_size== '882':
            latent_post = latent_post.view(1, 20, 8 * 8 * 2)
        
        latent_post = latent_post[0]  # only keep 0-19

        # 2. Input Branch
        latent_prior = self.encoder_x(x3d)  # --> torch.Size([20, 1024])

        # 3. KL Branch
        mu_y, logvar_y, dist_post = self.dist(latent_post)  # torch.Size([20, 1024]) -->
        mu_x, logvar_x, dist_prior = self.dist(latent_prior)     
        
        # 4. Loss
        latent_loss = torch.sum(self.kl_divergence(dist_post, dist_prior))
           
        # 5 Reconstruction
        z_noise_post = self.reparametrize(mu_y, logvar_y)  # --> torch.Size([20, 1024]) 
        z_noise_post = z_noise_post.unsqueeze(0)  # torch.Size([20, 1024]) --> torch.Size([1, 20, 1024])
        
        # torch.Size([1, 20, 1024]) --> torch.Size([1, 20, 16, 16, 4])
        if self.l_size == '32322':
            z_noise_post = z_noise_post.view(1, 20, 32, 32, 8)
        elif self.l_size== '16162':
            z_noise_post = z_noise_post.view(1, 20, 16, 16, 4)
        elif self.l_size== '882':
            z_noise_post = z_noise_post.view(1, 20, 8, 8, 2)
        
        recons_logit = self.encoder_y.forward_prior(z_noise_post)  # --> torch.Size([1, 20, 128, 128, 16])

        return recons_logit, recons_loss_post, vq_loss_post, latent_loss

    
    def forward_test(self, x3d):
        '''
        input:
            x3d: torch.Size([1, 128, 128, 128, 16])
        '''
        # 2. Input Branch
        latent_prior = self.encoder_x(x3d)  # --> torch.Size([20, 1024])

        # 3. KL Branch
        mu_x, logvar_x, dist_prior = self.dist(latent_prior)     
        
        # 5 Reconstruction
        z_noise_prior = self.reparametrize(mu_x, logvar_x)  # --> torch.Size([20, 1024]) 
        z_noise_prior = z_noise_prior.unsqueeze(0)  # torch.Size([20, 1024]) --> torch.Size([1, 20, 1024])
        
        # torch.Size([1, 20, 1024]) --> torch.Size([1, 20, 16, 16, 4])
        if self.l_size == '32322':
            z_noise_prior = z_noise_prior.view(1, 20, 32, 32, 8)
        elif self.l_size== '16162':
            z_noise_prior = z_noise_prior.view(1, 20, 16, 16, 4)
        elif self.l_size== '882':
            z_noise_prior = z_noise_prior.view(1, 20, 8, 8, 2)
        
        recons_logit = self.encoder_y.forward_prior(z_noise_prior)  # --> torch.Size([1, 20, 128, 128, 16])

        return recons_logit
        
