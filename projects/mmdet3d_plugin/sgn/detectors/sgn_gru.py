from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from ..utils import GridMask
from mmdet.models import builder

@DETECTORS.register_module()
class SGNGRU(MVXTwoStageDetector):
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
                 gru_head=None,
                 ):

        super(SGNGRU,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        
        if gru_head:
            self.gru_head = builder.build_head(gru_head)
     
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        
        self.only_occ = occupancy

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""

        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            if self.use_grid_mask is None:
                use_grid_mask = self.use_grid_mask
            if self.use_grid_mask:
                img = self.grid_mask(img)
                
            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        '''
        img_feats:
            [
                torch.Size([1, 5, 512, 47, 153])
                torch.Size([1, 5, 1024, 24, 77])
                torch.Size([1, 5, 2048, 12, 39])
            ]
        '''
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        
        '''
        img_feats_reshaped:
            [
                torch.Size([1, 5, 128, 47, 153])
                torch.Size([1, 5, 128, 24, 77])
                torch.Size([1, 5, 128, 12, 39])
                torch.Size([1, 5, 128, 6, 20])
            ]
        '''

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

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

        len_queue = img.size(1)
        batch_size = img.shape[0]
        img_W = img.shape[5]
        img_H = img.shape[4]
        
        img_metas = [each[len_queue-1] for each in img_metas]
        img = img[:, -1, ...]
        if self.only_occ:
            img_feats = None
        else:
            img_feats = self.extract_feat(img=img) 
        
        out = self.gru_head(img_feats)
        '''
        [
            torch.Size([1, 128, 47, 153])
            torch.Size([1, 128, 24, 77])
            torch.Size([1, 128, 12, 39])
            torch.Size([1, 128, 6, 20])
        ]
        '''
        out = [out[1].unsqueeze(1)]

        losses = dict()

        losses_pts = self.forward_pts_train(out, img_metas, target)
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
        if self.only_occ:
            img_feats = None
        else:
            img_feats = self.extract_feat(img=img)  
        
        out = self.gru_head(img_feats)
        '''
        [
            torch.Size([1, 128, 47, 153])
            torch.Size([1, 128, 24, 77])
            torch.Size([1, 128, 12, 39])
            torch.Size([1, 128, 6, 20])
        ]
        '''
        out = [out[1].unsqueeze(1)]

        
        outs = self.pts_bbox_head(out, img_metas, target)
        completion_results = self.pts_bbox_head.validation_step(outs, target, img_metas)

        return completion_results
