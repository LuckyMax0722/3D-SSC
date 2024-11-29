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
            
            # 1. Branch 
            self.target_backbone = builder.build_backbone(target_backbone_dict)
            self.tpv_backbone = builder.build_backbone(tpv_backbone_dict)
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
    
                latent_loss = torch.mean(torch.stack(latent_losses))
                
                print("Latent Loss:", latent_loss.item())
                print("Reparameterized z_noise_post shapes:", [z.shape for z in z_noises_post])

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
            voxel_global_feats:
                output: list -->
                [
                    torch.Size([1, 128, 128, 128])
                    torch.Size([1, 128, 128, 16])
                    torch.Size([1, 128, 128, 16])
                ]
            '''
            
            self.get_processed_feats(tpv_global_feats, target_feats)
            
            voxel_local_feats = self.voxel_backbone(x3d)
            '''
            voxel_local_feats:
                torch.Size([1, 128, 128, 128, 16])
            '''

            
            # KL Part
            
            
            
            
            weights = self.combine_coeff(voxel_local_feats)

            out_feats = voxel_local_feats * weights[:, 0:1, ...] + tpv_global_feats[0] * weights[:, 1:2, ...] + \
                tpv_global_feats[1] * weights[:, 2:3, ...] + tpv_global_feats[2] * weights[:, 3:4, ...]
            '''
            out_feats:
                torch.Size([1, 128, 128, 128, 16])
            '''
            
            
            print(out_feats.size())
            


            
