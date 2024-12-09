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
_temporal_ = [-12, -9, -6, -3]
point_cloud_range = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]

_sem_scal_loss_ = True
_geo_scal_loss_ = True
_depthmodel_= 'msnet3d'


model = dict(
   type='SGN',
   pretrained=dict(img=CONF.PATH.CHECKPOINT_RESNET50),
   img_backbone=dict(
       type='ResNet',
       depth=50,
       num_stages=4,
       out_indices=(2,),
       frozen_stages=1,
       norm_cfg=dict(type='BN', requires_grad=False),
       norm_eval=True,
       style='pytorch'),
   img_neck=dict(
       type='FPN',
       in_channels=[1024],
       out_channels=_dim_,
       start_level=0,
       add_extra_convs='on_output',
       num_outs=1,
       relu_before_extra_convs=True),
   pts_bbox_head=dict(
       type='SGNHeadOne',
       bev_h=128,
       bev_w=128,
       bev_z=16,
       embed_dims=_dim_,
       pts_header_dict=dict(
           type='SGNHeadOcc',
           point_cloud_range=point_cloud_range,
           spatial_shape=[256,256,32],
           guidance=True,
           nbr_classes=1),
       
       latent_header_dict=dict(
            type='LatentNet',
            embed_dims=CONF.LATENTNET.V6_geo_feat_dim,
            spatial_shape=[128,128,16],
            
            target_backbone_dict=dict(
                type='AutoEncoderGroupSkip',
                num_class=20,
                geo_feat_channels=CONF.LATENTNET.V6_geo_feat_dim,
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
            
            tpv_backbone_dict=dict(
                type='TPVGlobalAggregator',
                embed_dims=CONF.LATENTNET.V6_geo_feat_dim,
                split=CONF.LATENTNET.V6_split,
                grid_size=[128,128,16],
                
                # tpv_encoder_backbone_dict=dict(
                #     type='Swin',
                #     embed_dims=96, # 96
                #     depths=[2, 2, 6, 2],
                #     num_heads=[3, 6, 12, 24],
                #     window_size=7,
                #     mlp_ratio=4,
                #     in_channels=CONF.LATENTNET.V6_geo_feat_dim,
                #     patch_size=4,
                #     strides=[1,2,2,2],
                #     frozen_stages=-1,
                #     qkv_bias=True,
                #     qk_scale=None,
                #     drop_rate=0.,
                #     attn_drop_rate=0.,
                #     drop_path_rate=0.2,
                #     patch_norm=True,
                #     out_indices=[1,2,3],
                #     with_cp=False,
                #     convert_weights=True,
                #     init_cfg=dict(
                #         type='Pretrained',
                #         checkpoint=CONF.LATENTNET.V6_swin_pretrain),
                #         ),
                # tpv_encoder_neck=dict(
                #     type='GeneralizedLSSFPN',
                #     in_channels=[192, 384, 768],
                #     out_channels=CONF.LATENTNET.V6_geo_feat_dim,
                #     start_level=0,
                #     num_outs=3,
                #     norm_cfg=dict(
                #         type='BN2d',
                #         requires_grad=True,
                #         track_running_stats=False),
                #     act_cfg=dict(
                #         type='ReLU',
                #         inplace=True),
                #     upsample_cfg=dict(
                #         mode='bilinear',
                #         align_corners=False),
                # ),
            ),    
            

            # voxel_backbone_dict=dict(
            #     type='LocalAggregator',
            #     local_encoder_backbone=dict(
            #         type='CustomResNet3D',
            #         numC_input=128,
            #         num_layer=[2, 2, 2],
            #         num_channels=[128, 128, 128],
            #         stride=[1, 2, 2]),
            #     local_encoder_neck=dict(
            #         type='GeneralizedLSSFPN',
            #         in_channels=[128, 128, 128],
            #         out_channels=_dim_,
            #         start_level=0,
            #         num_outs=3,
            #         norm_cfg=dict(
            #             type='GN', 
            #             num_groups=32,
            #             requires_grad=True),
            #         conv_cfg=dict(type='Conv3d'),
            #         act_cfg=dict(
            #             type='ReLU',
            #             inplace=True),
            #         upsample_cfg=dict(
            #             mode='trilinear',
            #             align_corners=False)
            #     ),
            # ),

        ),

       CE_ssc_loss=True,
       geo_scal_loss=_geo_scal_loss_,
       sem_scal_loss=_sem_scal_loss_,
       scale_2d_list=[16]  # 16
    ),
   
   train_cfg=dict(pts=dict(
       grid_size=[512, 512, 1],
       voxel_size=voxel_size,
       point_cloud_range=point_cloud_range,
       out_size_factor=4)))


dataset_type = 'SemanticKittiDataset'
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
