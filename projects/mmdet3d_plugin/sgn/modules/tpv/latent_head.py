import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from mmcv.runner import BaseModule
from torch.distributions import Normal, Independent, kl
from mmdet.models import HEADS, builder


@HEADS.register_module()
class LatentHead(BaseModule):
        def __init__(self,
                     embed_dims,
                     spatial_shape,
                     use_post,
                     use_tpv_aggregator
                     ):
            super().__init__()
            self.spatial_shape = spatial_shape
            self.embed_dims = embed_dims
            self.use_post = use_post
            
            if use_tpv_aggregator:
                self.version = 'w_tpv_agg'
            else:
                self.version = 'wo_tpv_agg'
            
            self.use_tpv_aggregator = use_tpv_aggregator
              
            # 2. KL
            self.fc_post_1 = nn.Linear(embed_dims * 2, embed_dims)
            self.fc_post_2 = nn.Linear(embed_dims * 2, embed_dims)
            
            self.fc_prior_1 = nn.Linear(embed_dims, embed_dims)
            self.fc_prior_2 = nn.Linear(embed_dims, embed_dims)
            
        
        def dist_post(self, x):
            mu = self.fc_post_1(x)
            logvar = self.fc_post_2(x)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            
            return [mu, logvar, dist]
        
        def dist_prior(self, x):
            mu = self.fc_prior_1(x)
            logvar = self.fc_prior_2(x)
            dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
            
            return [mu, logvar, dist]
        
        def kl_divergence(self, posterior_latent_space, prior_latent_space):
            kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
            return kl_div
        
        def reparametrize(self, mu, logvar):
            std = logvar.mul(0.5).exp_()
            eps = torch.cuda.FloatTensor(std.size()).normal_()
            return eps.mul(std).add_(mu)
        
        def process_feats(self, input_tensor):
            input_tensor = input_tensor.squeeze(0)  
            # torch.Size([1, 128, 128, 128]) --> torch.Size([128, 128, 128])
            
            input_tensor = input_tensor.view(input_tensor.shape[0], -1)
            # torch.Size([128, 128, 128]) --> torch.Size([128, 128 * 128])
            
            input_tensor = input_tensor.permute(1, 0)  
            #[128, 128*128] -> [128, 128*128]

            return input_tensor
            
        def process_dim(self, idx, input_tensor, version):
            input_tensor = input_tensor.permute(1, 0)

            if version == 'wo_tpv_agg':
                if idx == 0:
                    input_tensor = input_tensor.view(self.embed_dims, self.spatial_shape[0], self.spatial_shape[1]).unsqueeze(-1)
                elif idx == 1:
                    input_tensor = input_tensor.view(self.embed_dims, self.spatial_shape[1], self.spatial_shape[2]).unsqueeze(1)
                elif idx == 2:
                    input_tensor = input_tensor.view(self.embed_dims, self.spatial_shape[0], self.spatial_shape[2]).unsqueeze(2)
            
            elif version == 'w_tpv_agg':
                if idx == 0:
                    input_tensor = input_tensor.view(self.embed_dims, self.spatial_shape[0], self.spatial_shape[1])
                elif idx == 1:
                    input_tensor = input_tensor.view(self.embed_dims, self.spatial_shape[1], self.spatial_shape[2])
                elif idx == 2:
                    input_tensor = input_tensor.view(self.embed_dims, self.spatial_shape[0], self.spatial_shape[2])
            
            return input_tensor.unsqueeze(0)
         
        def get_processed_feats(self, x3d, target):
            '''
            input:
                target: 
                    [
                        torch.Size([1, 128, 128, 128])
                        torch.Size([1, 128, 128, 16])
                        torch.Size([1, 128, 128, 16])
                    ]
                    
                    or
                    
                    None
                    
                x3d: 
                    [
                        torch.Size([1, 128, 128, 128])
                        torch.Size([1, 128, 128, 16])
                        torch.Size([1, 128, 128, 16])
                    ]
            '''
            
            if target:
                processed_x3d = [self.process_feats(tensor) for tensor in x3d]
                processed_target = [self.process_feats(tensor) for tensor in target]
                
                post_feats = [torch.cat([t1, t2], dim=1) for t1, t2 in zip(processed_x3d, processed_target)]
                # [*, 128] cat [*, 128] --> [*, 256]
                
                post_feats = [self.dist_post(x) for x in post_feats]
                # [[mu, logvar, dist], [mu, logvar, dist], [mu, logvar, dist]]
                prior_feats = [self.dist_prior(x) for x in processed_x3d]
                # [[mu, logvar, dist], [mu, logvar, dist], [mu, logvar, dist]]
                
                latent_losses = []
                z_noises_post = []
                z_noises_prior = []
                
                for post, prior in zip(post_feats, prior_feats):
                    mu_post, logvar_post, dist_post = post
                    mu_prior, logvar_prior, dist_prior = prior
                    
                    z_noise_prior = self.reparametrize(mu_prior, logvar_prior)
                    z_noises_prior.append(z_noise_prior)
                    
                    z_noise_post = self.reparametrize(mu_post, logvar_post)
                    z_noises_post.append(z_noise_post)
                    
                    kl_loss = self.kl_divergence(dist_post, dist_prior)
                    latent_losses.append(torch.mean(kl_loss))

                # [torch.Size([16384, 128]), torch.Size([2048, 128]), torch.Size([2048, 128])]
                z_noises_post = [self.process_dim(i, x, self.version) for i, x in enumerate(z_noises_post)]
                # [torch.Size([1, 128, 128, 128, 1]), torch.Size([1, 128, 1, 128, 16]), torch.Size([1, 128, 128, 1, 16])]
                
                # [torch.Size([16384, 128]), torch.Size([2048, 128]), torch.Size([2048, 128])]
                z_noises_prior = [self.process_dim(i, x, self.version) for i, x in enumerate(z_noises_prior)]
                # [torch.Size([1, 128, 128, 128, 1]), torch.Size([1, 128, 1, 128, 16]), torch.Size([1, 128, 128, 1, 16])]
                
                latent_loss = torch.mean(torch.stack(latent_losses))

                if self.use_post:
                    return latent_loss, z_noises_post
                else:
                    return latent_loss, z_noises_prior
            
            else:
                processed_x3d = [self.process_feats(tensor) for tensor in x3d]

                prior_feats = [self.dist_prior(x) for x in processed_x3d]
                # [[mu, logvar, dist], [mu, logvar, dist], [mu, logvar, dist]]
                
                z_noises_prior = []

                for prior in prior_feats:
                    mu_prior, logvar_prior, dist_prior = prior
                    
                    z_noise_prior = self.reparametrize(mu_prior, logvar_prior)
                    z_noises_prior.append(z_noise_prior)

                # [torch.Size([16384, 128]), torch.Size([2048, 128]), torch.Size([2048, 128])]
                z_noises_prior = [self.process_dim(i, x, self.version) for i, x in enumerate(z_noises_prior)]
                # [torch.Size([1, 128, 128, 128, 1]), torch.Size([1, 128, 1, 128, 16]), torch.Size([1, 128, 128, 1, 16])]
                   
                return z_noises_prior

        def get_tpv_aggregator(self, tpv_list):
            """
            tpv_list[0]: bs, c, h*w
            tpv_list[1]: bs, c, z*h
            tpv_list[2]: bs, c, w*z
            """
            tpv_h, tpv_w, tpv_z = self.spatial_shape[0], self.spatial_shape[1], self.spatial_shape[2]
            tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]

            tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, tpv_z)
            tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 2, 3).expand(-1, -1, tpv_w, -1, -1)
            tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, tpv_h, -1)
        
            fused = tpv_hw + tpv_zh + tpv_wz  # [bs, c, w, h, z]
            fused = fused.permute(0, 1, 3, 2, 4)

            return fused
        
        def forward_train(self, input_feats, target_feats):
            '''
            input:

            input_feats:
                input_feats = [tpv_hw, tpv_zh, tpv_wz], where sizes are -->
                [
                    torch.Size([1, 128, 128, 128])
                    torch.Size([1, 128, 128, 16])
                    torch.Size([1, 128, 128, 16])
                ]
            
            target_feats:
                target_feats = [xy_feat, xz_feat, yz_feat], where sizes are -->
                [
                    torch.Size([1, 128, 128, 128])
                    torch.Size([1, 128, 128, 16])
                    torch.Size([1, 128, 128, 16])
                ]
            '''
            
         
            # KL Part
            latent_loss, z = self.get_processed_feats(input_feats, target_feats)
            
            if self.use_tpv_aggregator:
                '''
                z:
                    output: list -->
                    [
                        torch.Size([1, 128, 128, 128])
                        torch.Size([1, 128, 128, 16])
                        torch.Size([1, 128, 128, 16])
                    ]
                '''
                z = self.get_tpv_aggregator(z)
                
                return latent_loss, z
            
            else:
                '''
                z:
                    output: list -->
                    [
                        torch.Size([1, 128, 128, 128, 1])
                        torch.Size([1, 128, 1, 128, 16])
                        torch.Size([1, 128, 128, 1, 16])
                    ]
                '''
                
                return latent_loss, z
        
        def forward_test(self, input_feats, target_feats):
            '''
            input:

            input_feats:
                input_feats = [tpv_hw, tpv_zh, tpv_wz], where sizes are -->
                [
                    torch.Size([1, 128, 128, 128])
                    torch.Size([1, 128, 128, 16])
                    torch.Size([1, 128, 128, 16])
                ]
            
            target_feats:
                target_feats = None
            '''
            
         
            # KL Part
            z = self.get_processed_feats(input_feats, None)
            
            if self.use_tpv_aggregator:
                '''
                z:
                    output: list -->
                    [
                        torch.Size([1, 128, 128, 128])
                        torch.Size([1, 128, 128, 16])
                        torch.Size([1, 128, 128, 16])
                    ]
                '''
                z = self.get_tpv_aggregator(z)
                
                return z
            
            else:
                '''
                z:
                    output: list -->
                    [
                        torch.Size([1, 128, 128, 128, 1])
                        torch.Size([1, 128, 1, 128, 16])
                        torch.Size([1, 128, 128, 1, 16])
                    ]
                '''
                
                return z
            
            

