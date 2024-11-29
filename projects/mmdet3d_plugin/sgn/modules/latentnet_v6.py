import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from mmcv.runner import BaseModule
from torch.distributions import Normal, Independent, kl
from mmdet.models import HEADS, builder

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF

from .latent_v6 import *

@HEADS.register_module()
class LatentNet(BaseModule):
        def __init__(self,
                     embed_dims,
                     spatial_shape,
                     target_backbone_dict=None,
                     tpv_backbone_dict=None,
                     voxel_backbone_dict=None
                     ):
            super().__init__()
            self.spatial_shape = spatial_shape
            self.embed_dims = embed_dims
            self.voxel_backbone_dict = voxel_backbone_dict
            
            # 1. Branch 
            self.target_backbone = builder.build_backbone(target_backbone_dict)
            self.tpv_backbone = builder.build_backbone(tpv_backbone_dict)
            
            if voxel_backbone_dict:
                self.voxel_backbone = builder.build_backbone(voxel_backbone_dict)
            
            self.combine_coeff = nn.Sequential(
                nn.Conv3d(embed_dims, 4, kernel_size=1, bias=False),
                nn.Softmax(dim=1)
            )
            
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
            
        def process_dim(self, idx, input_tensor):
            input_tensor = input_tensor.permute(1, 0)

            if idx == 0:
                input_tensor = input_tensor.view(self.embed_dims, self.spatial_shape[0], self.spatial_shape[1]).unsqueeze(-1)
            elif idx == 1:
                input_tensor = input_tensor.view(self.embed_dims, self.spatial_shape[1], self.spatial_shape[2]).unsqueeze(1)
            elif idx == 2:
                input_tensor = input_tensor.view(self.embed_dims, self.spatial_shape[0], self.spatial_shape[2]).unsqueeze(2)
            
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

                for post, prior in zip(post_feats, prior_feats):
                    mu_post, logvar_post, dist_post = post
                    mu_prior, logvar_prior, dist_prior = prior
                    
                    z_noise_post = self.reparametrize(mu_post, logvar_post)
                    z_noises_post.append(z_noise_post)
                    
                    kl_loss = self.kl_divergence(dist_post, dist_prior)
                    latent_losses.append(torch.mean(kl_loss))

                # [torch.Size([16384, 128]), torch.Size([2048, 128]), torch.Size([2048, 128])]
                z_noises_post = [self.process_dim(i, x) for i, x in enumerate(z_noises_post)]
                # [torch.Size([1, 128, 128, 128, 1]), torch.Size([1, 128, 1, 128, 16]), torch.Size([1, 128, 128, 1, 16])]
                
                latent_loss = torch.mean(torch.stack(latent_losses))
            
                return latent_loss, z_noises_post
            
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
                z_noises_prior = [self.process_dim(i, x) for i, x in enumerate(z_noises_prior)]
                # [torch.Size([1, 128, 128, 128, 1]), torch.Size([1, 128, 1, 128, 16]), torch.Size([1, 128, 128, 1, 16])]
                   
                return z_noises_prior

        def forward_train(self, x3d, target):
            '''
            input:
                target: torch.Size([1, 128, 128, 16])
                x3d: torch.Size([1, 128, 128, 128, 16])
            '''   
            
            target_feats = self.target_backbone(target.long(), None)
            '''
            target_feats:
                target_feats = [xy_feat, xz_feat, yz_feat], where sizes are -->
                [
                    torch.Size([1, 128, 128, 128])
                    torch.Size([1, 128, 128, 16])
                    torch.Size([1, 128, 128, 16])
                ]
            '''
            
            tpv_global_feats = self.tpv_backbone(x3d)
            '''
            tpv_global_feats:
                output: list -->
                [
                    torch.Size([1, 128, 128, 128])
                    torch.Size([1, 128, 128, 16])
                    torch.Size([1, 128, 128, 16])
                ]
            '''
            # KL Part
            latent_loss, z = self.get_processed_feats(tpv_global_feats, target_feats)
            
            '''
            z:
                output: list -->
                [
                    torch.Size([1, 128, 128, 128, 1])
                    torch.Size([1, 128, 1, 128, 16])
                    torch.Size([1, 128, 128, 1, 16])
                ]
            '''
            
            if self.voxel_backbone_dict:
                voxel_local_feats = self.voxel_backbone(x3d)
            else:
                voxel_local_feats = x3d
            '''
            voxel_local_feats:
                torch.Size([1, 128, 128, 128, 16])
            '''

 
            weights = self.combine_coeff(voxel_local_feats)

            out_feats = voxel_local_feats * weights[:, 0:1, ...] + z[0] * weights[:, 1:2, ...] + \
                z[1] * weights[:, 2:3, ...] + z[2] * weights[:, 3:4, ...]
            '''
            out_feats:
                torch.Size([1, 128, 128, 128, 16])
            '''
            
            
            return latent_loss, out_feats
        
        def forward_test(self, x3d, target):
            '''
            input:
                target: None
                x3d: torch.Size([1, 128, 128, 128, 16])
            '''   
            
            tpv_global_feats = self.tpv_backbone(x3d)
            '''
            tpv_global_feats:
                output: list -->
                [
                    torch.Size([1, 128, 128, 128])
                    torch.Size([1, 128, 128, 16])
                    torch.Size([1, 128, 128, 16])
                ]
            '''
            # KL Part
            z = self.get_processed_feats(tpv_global_feats, None)
            
            '''
            z:
                output: list -->
                [
                    torch.Size([1, 128, 128, 128, 1])
                    torch.Size([1, 128, 1, 128, 16])
                    torch.Size([1, 128, 128, 1, 16])
                ]
            '''
            
            if self.voxel_backbone_dict:
                voxel_local_feats = self.voxel_backbone(x3d)
            else:
                voxel_local_feats = x3d
            '''
            voxel_local_feats:
                torch.Size([1, 128, 128, 128, 16])
            '''

 
            weights = self.combine_coeff(voxel_local_feats)

            out_feats = voxel_local_feats * weights[:, 0:1, ...] + z[0] * weights[:, 1:2, ...] + \
                z[1] * weights[:, 2:3, ...] + z[2] * weights[:, 3:4, ...]
            '''
            out_feats:
                torch.Size([1, 128, 128, 128, 16])
            '''
            
            
            return out_feats
            


            
