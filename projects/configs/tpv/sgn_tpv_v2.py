import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from projects.configs.config import CONF

work_dir = ''
_base_ = [
    '../_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

_dim_ = 128

_labels_tag_ = 'labels'
_temporal_ = [-12, -9, -6, -3, 0]
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]

_sem_scal_loss_ = True
_geo_scal_loss_ = True
_depthmodel_= 'msnet3d'

tpv_h_ = 128
tpv_w_ = 128
tpv_z_ = 16
scale_h = 1
scale_w = 1
scale_z = 1
grid_size = [tpv_h_*scale_h, tpv_w_*scale_w, tpv_z_*scale_z]

_num_levels_=4
_num_cams_=1

_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2

num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
nbr_class = 20

model = dict(
    type='SGNTPV',
    use_grid_mask=True,
    pretrained=dict(img=CONF.PATH.CHECKPOINT_RESNET50),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        dcn=dict(
            type='DCNv2', 
            deform_groups=1, 
            fallback_on_stride=False), 
        stage_with_dcn=(False, False, True, True)
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    tpv_head=dict(
        type='TPVFormerHead',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        pc_range=point_cloud_range,
        num_feature_levels=_num_levels_,
        num_cams=_num_cams_,
        embed_dims=_dim_,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=tpv_h_,
            col_num_embed=tpv_w_),
        encoder=dict(
            type='TPVFormerEncoder',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_layers=2,
            pc_range=point_cloud_range,
            num_points_in_pillar=num_points_in_pillar,
            return_intermediate=False,
            transformerlayers=dict(
                type='TPVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TPVCrossViewHybridAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='TPVImageCrossAttention',
                        num_cams=_num_cams_,
                        pc_range=point_cloud_range,
                        embed_dims=_dim_,
                        tpv_h=tpv_h_,
                        tpv_w=tpv_w_,
                        tpv_z=tpv_z_,
                        deformable_attention=dict(
                            type='TPVMSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=num_points,
                            num_z_anchors=num_points_in_pillar,
                            num_levels=_num_levels_,
                            floor_sampling_offset=False,
                            tpv_h=tpv_h_,
                            tpv_w=tpv_w_,
                            tpv_z=tpv_z_,
                        ),
                    )
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')))
        ),

    target_head=dict(
        type='AutoEncoderGroupSkipV2',
        num_class=20,
        geo_feat_channels=_dim_,
        padding_mode='replicate',
        z_down=True,
        voxel_fea=False,
        pos=True,
        feat_channel_up=64,
        triplane=True,
        mlp_hidden_channels=128,
        mlp_hidden_layers=3,
        dataset='kitti',
    ),
    
    latent_head=dict(
        type='LatentHeadV2',
        embed_dims=_dim_,
        spatial_shape=[128,128,16],
        use_tpv_aggregator=CONF.TPV.USE_V3
    ),
    
    pts_bbox_head=dict(
       type='SGNHeadOneV2',
       bev_h=128,
       bev_w=128,
       bev_z=16,
       embed_dims=_dim_,
       pts_header_dict=dict(
           type='SGNHeadOccV2',
           point_cloud_range=point_cloud_range,
           spatial_shape=[256,256,32],
           guidance=True,
           nbr_classes=1),
       CE_ssc_loss=True,
       geo_scal_loss=_geo_scal_loss_,
       sem_scal_loss=_sem_scal_loss_,
       scale_2d_list=[16]  # 16
       ),
 
   train_cfg=dict(pts=dict(
       grid_size=[512, 512, 1],
       voxel_size=voxel_size,
       point_cloud_range=point_cloud_range,
       out_size_factor=4))

)


dataset_type = 'SemanticKittiDatasetV2'
data_root = CONF.PATH.DATA + '/'
file_client_args = dict(backend='disk')

data = dict(
   samples_per_gpu=1,
   workers_per_gpu=4,
   train=dict(
       type=dataset_type,
       split = "train",
       test_mode=False,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       temporal = _temporal_,
       labels_tag = _labels_tag_),
   val=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       temporal = _temporal_,
       labels_tag = _labels_tag_),
   test=dict(
       type=dataset_type,
       split = "val",
       test_mode=True,
       data_root=data_root,
       preprocess_root=data_root + 'dataset',
       eval_range = 51.2,
       depthmodel=_depthmodel_,
       temporal = _temporal_,
       labels_tag = _labels_tag_),
   shuffler_sampler=dict(type='DistributedGroupSampler'),
   nonshuffler_sampler=dict(type='DistributedSampler')
)
optimizer = dict(
   type='AdamW',
   lr=2e-4,
   weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
   policy='CosineAnnealing',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=1.0 / 3,
   min_lr_ratio=1e-3)
total_epochs = 48
evaluation = dict(interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
   interval=50,
   hooks=[
       dict(type='TextLoggerHook'),
       dict(type='TensorboardLoggerHook')
   ])

# checkpoint_config = None
checkpoint_config = dict(interval=1)
