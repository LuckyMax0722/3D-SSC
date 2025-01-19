import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmdet.models import HEADS


@HEADS.register_module()
class LatentHeadV2(BaseModule):
        def __init__(self,
                     embed_dims,
                     spatial_shape,
                     use_tpv_aggregator,
                     ):
            super().__init__()
            self.spatial_shape = spatial_shape
            self.embed_dims = embed_dims
            self.loss = nn.CosineEmbeddingLoss()
            
            if use_tpv_aggregator:
                self.version = 'w_tpv_agg'
            else:
                self.version = 'wo_tpv_agg'
            
            self.use_tpv_aggregator = use_tpv_aggregator
            
        
        def process_feats(self, input_tensor):
            #input_tensor = input_tensor.squeeze(0)  
            # torch.Size([1, 128, 128, 128]) --> torch.Size([128, 128, 128])
            
            input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)
            # torch.Size([1, 128, 128, 128]) --> torch.Size([1, 128 * 128 * 128])
            
            #input_tensor = input_tensor.permute(1, 0)  
            #[128, 128*128] -> [128, 128*128]

            return input_tensor
        
        def get_tpv_aggregator(self, tpv_list):
            """
            [
                torch.Size([1, 128, 128, 128])
                torch.Size([1, 128, 128, 16])
                torch.Size([1, 128, 128, 16])
            ]
            """
            tpv_list[1] = tpv_list[1].permute(0, 1, 3, 2)
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
                    torch.Size([1, 128, 16, 128])
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
            
            '''
            # cos_sim
            cos_sim_loss = 0
    
            for i, t in zip(input_feats, target_feats):
                i = self.process_feats(i)
                t = self.process_feats(t)
                
                # cosine_similarity
                cos_sim = F.cosine_similarity(i, t, dim=1)
                
                # Log_softmax
                log_softmax_scores = F.log_softmax(cos_sim, dim=-1)
                
                # Loss
                loss = -torch.mean(log_softmax_scores)
                
                cos_sim_loss = cos_sim_loss + loss
            '''

            processed_x3d = [self.process_feats(tensor) for tensor in input_feats]
            processed_target = [self.process_feats(tensor) for tensor in target_feats]
            
            label = torch.ones(1).to(target_feats[0].device)
            loss_latent = 0
            
            for t, x in zip(processed_x3d, processed_target):
                loss_latent = loss_latent + self.loss(t, x, label)
    
            
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
                input_feats = self.get_tpv_aggregator(input_feats)
                
                return loss_latent, input_feats
            
            else:
                '''
                Not implement
                '''
                
                raise NotImplementedError("LatentHeadV2 must use TPV Aggregator")
        
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
                input_feats = self.get_tpv_aggregator(input_feats)
                
                return input_feats
            
            else:
                '''
                Not implement
                '''
                
                raise NotImplementedError("LatentHeadV2 must use TPV Aggregator")

if __name__ == '__main__':
    target = [
        torch.randn(1, 128, 128, 128),
        torch.randn(1, 128, 128, 16),
        torch.randn(1, 128, 128, 16),
    ]

    x3d = [
        torch.randn(1, 128, 128, 128),
        torch.randn(1, 128, 128, 16),
        torch.randn(1, 128, 128, 16),
    ]

    
    
    loss = nn.CosineEmbeddingLoss()
    label = torch.ones(1)
    loss_latent = 0
    
    for t, x in zip(target, x3d):
        t = t.view(t.size(0), -1)
        x = x.view(x.size(0), -1)
        
        loss_latent = loss_latent + loss(t, x, label)
    
    print(loss_latent)