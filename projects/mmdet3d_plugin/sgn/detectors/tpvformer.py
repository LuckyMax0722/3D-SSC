from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS, builder
import warnings
from ...dataloader.grid_mask import GridMask
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

@DETECTORS.register_module()
class TPVFormer(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 tpv_head=None,
                 pretrained=None,
                 tpv_aggregator=None,
                 **kwargs,
                 ):

        super().__init__()

        if tpv_head:
            self.tpv_head = builder.build_head(tpv_head)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)
        if tpv_aggregator:
            self.tpv_aggregator = builder.build_head(tpv_aggregator)

        if pretrained is None:
            img_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
            
        if img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.img_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

    @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img, use_grid_mask=None):
        """Extract features of images."""

        B = img.size(0)
        if img is not None:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
            if use_grid_mask is None:
                use_grid_mask = self.use_grid_mask
            if use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if hasattr(self, 'img_neck'):
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        # out list
        # torch.Size([1, 5, 256, 47, 153])
        # torch.Size([1, 5, 256, 24, 77])
        # torch.Size([1, 5, 256, 12, 39])
        # torch.Size([1, 5, 256, 6, 20])

        '''
        !!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        
        img_feats_reshaped = [x[:, 0:1, :, :, :] for x in img_feats_reshaped]

        return img_feats_reshaped

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
                #points=None,
                img_metas=None,
                img=None,
                target=None
        ):
        """
        Forward training function.
        
        Input:
            img: torch.Size([1, 1, 5, 3, 370, 1220]) == ([bs, len_queue, 5, 3, H, W])
        """
        # 1. Img Preprocess
        batch_size = img.shape[0]
        len_queue = img.size(1)
        
        img = img[:, -1, ...]
        
        img_feats = self.extract_img_feat(img=img, use_grid_mask=self.use_grid_mask)
        outs = self.tpv_head(img_feats, img_metas)
        #outs = self.tpv_aggregator(outs, points)
        return outs
    
    def forward_test(self,
                     img_metas=None,
                     img=None,
                     target=None,
                      **kwargs):
        
        return None