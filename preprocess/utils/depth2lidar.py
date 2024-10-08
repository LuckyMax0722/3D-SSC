import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti_util
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from projects.configs.config import CONF

def project_disp_to_depth(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

class Depth2Lidar:
    def __init__(self, calib_dir, depth_dir, save_dir, max_high):
        self.calib_dir = calib_dir
        self.depth_dir = depth_dir
        self.save_dir = save_dir
        self.max_high = max_high

        assert os.path.isdir(self.depth_dir)
        assert os.path.isdir(self.calib_dir)

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def process_point_cloud(self):
        depths = [x for x in os.listdir(self.depth_dir) if x[-3:] == 'npy' and 'std' not in x]
        depths = sorted(depths)

        for fn in depths:
            predix = fn[:-4]
            # predix = fn[:-8]
            # calib_file = '{}/{}.txt'.format(args.calib_dir, predix)
            calib_file = '{}/{}.txt'.format(self.calib_dir, 'calib')
            calib = kitti_util.Calibration(calib_file)
            depth_map = np.load(self.depth_dir + '/' + fn)

            lidar = project_disp_to_depth(calib, depth_map, self.max_high)
            # pad 1 in the indensity dimension
            lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
            lidar = lidar.astype(np.float32)
            lidar.tofile('{}/{}.bin'.format(self.save_dir, predix))
            print('Finish Depth {}'.format(predix))



if __name__ == '__main__':
    sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', 
                 '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                 '20', '21']
    
    for seq in sequences:
        calib_dir = os.path.join(CONF.PATH.DATA_DATASETS_SEQUENCES, seq)
        depth_dir = os.path.join(CONF.PATH.DATA_DATASETS_MSNET3D_DEPTH, 'sequences', seq)
        save_dir = os.path.join(CONF.PATH.DATA_DATASETS_MSNET3D_LIDAR, 'sequences', seq)

        D2L = Depth2Lidar(
            calib_dir=calib_dir,
            depth_dir=depth_dir,
            save_dir=save_dir,
            max_high = 80
        )

        D2L.process_point_cloud()

    
