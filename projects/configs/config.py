import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/u/home/caoh/projects/MA_Jiachen/SGN'  # TODO: Change path to your SGN-dir

# data
CONF.PATH.DATA = '/u/home/caoh/datasets/SemanticKITTI'  # TODO: Change path to your SemanticKITTI Data-dir
CONF.PATH.DATA_DATASETS = os.path.join(CONF.PATH.DATA, 'dataset')
CONF.PATH.DATA_DATASETS_POSES = os.path.join(CONF.PATH.DATA_DATASETS, 'poses')
CONF.PATH.DATA_DATASETS_SEQUENCES = os.path.join(CONF.PATH.DATA_DATASETS, 'sequences')
CONF.PATH.DATA_DATASETS_MSNET3D_DEPTH = os.path.join(CONF.PATH.DATA_DATASETS, 'sequences_msnet3d_depth')
CONF.PATH.DATA_DATASETS_MSNET3D_LIDAR = os.path.join(CONF.PATH.DATA_DATASETS, 'sequences_msnet3d_lidar')

# sub file dir
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.BASE, 'output')
CONF.PATH.PREPROCESS = os.path.join(CONF.PATH.BASE, 'preprocess')
CONF.PATH.MOBILESTEREONET = os.path.join(CONF.PATH.PREPROCESS, 'mobilestereonet')
CONF.PATH.DEPTHANYTHING = os.path.join(CONF.PATH.PREPROCESS, 'depthanythingv2')

# config
CONF.PATH.SEMANTICKITTI_YAML = os.path.join(CONF.PATH.PREPROCESS, 'label', 'semantic-kitti.yaml')
CONF.PATH.SGN_CONFIG = os.path.join(CONF.PATH.BASE, 'projects/configs/sgn/sgn-T-one-stage-guidance.py')

# pre-train model
CONF.PATH.CHECKPOINT = os.path.join(CONF.PATH.BASE, 'ckpt')
CONF.PATH.CHECKPOINT_RESNET50 = os.path.join(CONF.PATH.CHECKPOINT, 'resnet50-19c8e357.pth')
CONF.PATH.CHECKPOINT_SGN = os.path.join(CONF.PATH.CHECKPOINT, 'sgn-t-epoch_25.pth')
#CONF.PATH.CHECKPOINT_SGN = os.path.join(CONF.PATH.OUTPUT, 'epoch_28.pth')

CONF.PATH.CHECKPOINT_MSN3D =  os.path.join(CONF.PATH.MOBILESTEREONET, 'MSNet3D_SF_DS_KITTI2015.ckpt')
CONF.PATH.CHECKPOINT_DA = os.path.join(CONF.PATH.DEPTHANYTHING, 'ckpt', 'depth_anything_v2_metric_vkitti_vitl.pth')

#semantiKITTI config
CONF.KITTI = EasyDict()

CONF.KITTI.POINT_CLOUD_RANGE = [0, -25.6, -2.0, 51.2, 25.6, 4.4]
CONF.KITTI.VOXEL_SIZE = [0.2, 0.2, 0.2]
CONF.KITTI.GRID_SIZE = [256, 256, 32]

CONF.KITTI.POINT_ENCODING_TYPE = 'absolute_coordinates_encoding'
CONF.KITTI.POINT_FEATURE_LIST = ['x', 'y', 'z', 'intensity']
CONF.KITTI.NUM_POINT_FEATURES= 4

# LatenNet config
CONF.LATENTNET = EasyDict()

# TODO: activate or deactivate KL part
CONF.LATENTNET.USE_V1 = False
CONF.LATENTNET.USE_V2 = False
CONF.LATENTNET.USE_V3 = False
CONF.LATENTNET.USE_V3_1 = False
CONF.LATENTNET.USE_V3_2 = False

CONF.LATENTNET.USE_V4 = False

CONF.LATENTNET.USE_V5 = False
CONF.LATENTNET.USE_V5_alpha = 0.1

CONF.LATENTNET.USE_V5_1 = False
CONF.LATENTNET.USE_V5_2 = False
CONF.LATENTNET.V5_1_pretrain = '/u/home/caoh/projects/MA_Jiachen/SGN/output/output_VQVAE_20/checkpoints/vqvae-epoch=34-val_loss=0.18.ckpt'

CONF.LATENTNET.LATENT_DIM = 128

CONF.LATENTNET.USE_V6 = True
CONF.LATENTNET.V6_swin_pretrain = '/u/home/caoh/projects/MA_Jiachen/SGN/ckpt/swin_tiny_patch4_window7_224.pth'
CONF.LATENTNET.USE_V6_1 = False

if CONF.LATENTNET.USE_V6:
    CONF.LATENTNET.V6_geo_feat_dim = 128
    CONF.LATENTNET.V6_split = [8,8,8]
elif CONF.LATENTNET.USE_V6_1:
    CONF.LATENTNET.V6_geo_feat_dim = 64
    CONF.LATENTNET.V6_split = [16,16,16]
    
# TODO: activate or deactivate Image+Depth part
CONF.FUSION = EasyDict()
CONF.FUSION.USE_V1 = False


# TODO: activate or deactivate Full Scale
CONF.FULL_SCALE = EasyDict()
CONF.FULL_SCALE.USE_V1 = False
CONF.FULL_SCALE.USE_V2 = False
CONF.FULL_SCALE.USE_V3 = False


# TODO: activate or deactivate Transformer part
CONF.TRANSFORMER = EasyDict()
CONF.TRANSFORMER.DIM = 128
CONF.TRANSFORMER.H = 24
CONF.TRANSFORMER.W = 77
CONF.TRANSFORMER.TCA = False  # TemporalCrossAttention

# TODO: activate or deactivate Uncertainty part
CONF.UNCERTAINTY = EasyDict()
CONF.UNCERTAINTY.USE_V1 = False

# TODO: activate or deactivate TPV
CONF.TPV = EasyDict()
CONF.TPV.USE_V1 = False

# PVRCNN config
CONF.PVRCNN = EasyDict()
CONF.PVRCNN.NUM_BEV_FEATURES = 128
CONF.PVRCNN.LAYER_NUMS = [5, 5]
CONF.PVRCNN.LAYER_STRIDES = [1, 2]
CONF.PVRCNN.NUM_FILTERS = [128, 256]
CONF.PVRCNN.UPSAMPLE_STRIDES = [1, 2]
CONF.PVRCNN.NUM_UPSAMPLE_FILTERS = [256, 256]

# VP2P config
CONF.VP2P = EasyDict()
CONF.VP2P.INPUT_PT_NUM = 40960

# Scene Scale Diffusion config
CONF.SSD = EasyDict()
CONF.SSD.init_size = 16
CONF.SSD.num_classes = 20  # 20 for 5.1 21 for 5.2
CONF.SSD.vq_size = 100
CONF.SSD.l_size = '16162'
CONF.SSD.l_attention = True
CONF.SSD.out_dim = 1024

# SGN para
CONF.SGN = EasyDict()
CONF.SGN.embed_dims = 128
CONF.SGN.class_num = 20