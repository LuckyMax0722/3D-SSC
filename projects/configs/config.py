import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/media/max/GAME/MA/SGN'  # TODO: Change path to your SGN-dir


CONF.PATH.DATA = '/media/max/GAME/MA/datasets/SemanticKITTI'  # TODO: Change path to your SemanticKITTI Data-dir
CONF.PATH.DATA_DATASETS = os.path.join(CONF.PATH.DATA, 'dataset')
CONF.PATH.DATA_DATASETS_POSES = os.path.join(CONF.PATH.DATA_DATASETS, 'poses')
CONF.PATH.DATA_DATASETS_SEQUENCES = os.path.join(CONF.PATH.DATA_DATASETS, 'sequences')
CONF.PATH.DATA_DATASETS_MSNET3D_DEPTH = os.path.join(CONF.PATH.DATA_DATASETS, 'sequences_msnet3d_depth')
CONF.PATH.DATA_DATASETS_MSNET3D_LIDAR = os.path.join(CONF.PATH.DATA_DATASETS, 'sequences_msnet3d_lidar')


CONF.PATH.LOG = os.path.join(CONF.PATH.BASE, 'log')

CONF.PATH.PREPROCESS = os.path.join(CONF.PATH.BASE, 'preprocess')
CONF.PATH.MOBILESTEREONET = os.path.join(CONF.PATH.PREPROCESS, 'mobilestereonet')
CONF.PATH.DEPTHANYTHING = os.path.join(CONF.PATH.PREPROCESS, 'depthanythingv2')

CONF.PATH.SEMANTICKITTI_YAML = os.path.join(CONF.PATH.PREPROCESS, 'label', 'semantic-kitti.yaml')


