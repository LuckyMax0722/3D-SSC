import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/media/max/GAME/MA/SGN'  # TODO: Change path to your SGN-dir

# data
CONF.PATH.DATA = '/media/max/GAME/MA/datasets/SemanticKITTI'  # TODO: Change path to your SemanticKITTI Data-dir
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