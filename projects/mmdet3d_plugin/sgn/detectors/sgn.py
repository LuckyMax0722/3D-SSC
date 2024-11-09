from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import torch


import torch.nn as nn
from .. import builder

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF

@DETECTORS.register_module()
class SGN(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 occupancy=False,
                 ):

        super(SGN,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.only_occ = occupancy
        
        # KL Part
        # use KL part or not
        if CONF.LATENTNET.USE_V1:
            from ..modules.latentnet_v1 import LatentNet
            self.latent = LatentNet()
        elif CONF.LATENTNET.USE_V2:
            from ..modules.latentnet_v2 import LatentNet
            self.latent = LatentNet()
        elif CONF.FUSION.USE_V1:
            self.Image_decoder_block1 = nn.Sequential(  
                nn.Conv2d(4, 3, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
            )
        elif CONF.LATENTNET.USE_V3:
            from ..modules.latentnet_v3 import LatentNet
            self.latent = LatentNet()
        elif CONF.TRANSFORMER.TCA:
            from ..modules.tca import TemporalCrossAttention
            self.tca = TemporalCrossAttention(
                in_channels=CONF.TRANSFORMER.DIM, 
                feature_dim=CONF.TRANSFORMER.DIM, 
                height=CONF.TRANSFORMER.H, 
                width=CONF.TRANSFORMER.W
            )
            
            

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""

        B = img.size(0)
        
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            img_feats = self.img_backbone(img)

            # bs = 1
            # img_head size torch.Size([5, 1024, 24, 77])

            # img_feats is a tuple
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
                
        else:
            return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            
            BN, C, H, W = img_feat.size()
            # bs = 1
            # img_neck size torch.Size([5, 128, 24, 77])
            
            # len_queue = None
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        # torch.Size([1, 5, 128, 24, 77])
        return img_feats_reshaped


    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        
        # bs = 1
        # img size torch.Size([1, 1, 5, 3, 370, 1220]) None None
        # img size torch.Size([bs, len_queue, 5, 3, H, W])
        
        B = img.size(0)
        
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          img_feats, 
                          img_metas,
                          target):
        """Forward function'
        """
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        losses = self.pts_bbox_head.training_step(outs, target, img_metas)
        return losses

    def forward_kl_train(self,
                          img_feats, 
                          img_metas,
                          target):
        """Forward function'
        """
        losses = self.latent.forward_train(img_feats, img_metas, target)
        return losses
    
    def forward_kl_test(self,
                          img_feats, 
                          img_metas,
                          target):
        """Forward function'
        """
        self.latent.forward_test(img_feats, img_metas, target)
    
    def forward_kl_v3_test(self,
                          img_feats, 
                          img_metas,
                          target):
        """Forward function'
        """
 
        return self.latent.forward_test(img_feats, img_metas, target)

        
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      target=None):
        """Forward training function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Losses of different branches.
        """

        # bs = 1
        # img size torch.Size([1, 1, 5, 3, 370, 1220])
        # img size torch.Size([bs, len_queue, 5, 3, H, W])
        
        # bs = 2
        # img size torch.Size([2, 1, 5, 3, 370, 1220])
        
        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]
        
        img_metas = [each[len_queue-1] for each in img_metas]
        
        # bs = 1
        # img size torch.Size([1, 5, 3, 370, 1220])
        # img size torch.Size([bs, 5, 3, H, W])
        img = img[:, -1, ...]  
        
        img_metas[0]['mode'] = 'train'
        
        losses = dict()
        
        if CONF.FUSION.USE_V1:
            # Img + Depth Imag
            device = img.device
            
            depth = img_metas[0]['depth_tensor'].to(device)  # torch.Size([5, 1, 370, 1220])
            
            img = torch.cat((img[0], depth), dim=1)  # torch.Size([5, 4, 370, 1220])
            
            img = self.Image_decoder_block1(img)  # torch.Size([5, 3, 370, 1220])
            
            img = img.unsqueeze(0)  # torch.Size([1, 5, 3, 370, 1220])

        if CONF.LATENTNET.USE_V3 or CONF.LATENTNET.USE_V2:
            losses_latent, img = self.forward_kl_train(img, img_metas, target)
            losses.update(losses_latent)
        
        if self.only_occ:
            img_feats = None
        else:
            img_feats = self.extract_feat(img=img)  # List!!

        if CONF.TRANSFORMER.TCA:
            img_feats = self.tca(img_feats[0])  # [torch.Size([1, 1, 128, 24, 77])]       
        
        if CONF.LATENTNET.USE_V1:
            losses_latent = self.forward_kl_train(img_feats, img_metas, target)
            losses.update(losses_latent)
        
        losses_pts = self.forward_pts_train(img_feats, img_metas, target)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     img_metas=None,
                     img=None,
                     target=None,
                      **kwargs):
        """Forward testing function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor): Images of each sample with shape
                (batch, C, H, W). Defaults to None.
            target (torch.Tensor): ground-truth of semantic scene completion
                (batch, X_grids, Y_grids, Z_grids)
        Returns:
            dict: Completion result.
        """

        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]
        
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        
        img_metas[0]['mode'] = 'test'
        
        if CONF.FUSION.USE_V1:
            # Img + Depth Imag
            device = img.device
            
            depth = img_metas[0]['depth_tensor'].to(device)  # torch.Size([5, 1, 370, 1220])
            
            img = torch.cat((img[0], depth), dim=1)  # torch.Size([5, 4, 370, 1220])
            
            img = self.Image_decoder_block1(img)  # torch.Size([5, 3, 370, 1220])
            
            img = img.unsqueeze(0)  # torch.Size([1, 5, 3, 370, 1220])
            
        if CONF.LATENTNET.USE_V3 or CONF.LATENTNET.USE_V2:
            img = self.forward_kl_v3_test(img, img_metas, target)       

        if self.only_occ:
            img_feats = None
        else:
            img_feats = self.extract_feat(img=img)  
        
        if CONF.TRANSFORMER.TCA:
            img_feats = self.tca(img_feats[0])  # [torch.Size([1, 1, 128, 24, 77])]       
            
        if CONF.LATENTNET.USE_V1:
            self.forward_kl_test(img_feats, img_metas, target)
            
        outs = self.pts_bbox_head(img_feats, img_metas, target)
        completion_results = self.pts_bbox_head.validation_step(outs, target, img_metas)

        return completion_results
