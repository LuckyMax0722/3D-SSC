"""
Born out of Depth Anything V1 Issue 36
Make sure you have the necessary libraries installed.
Code by @1ssb

This script processes a set of images to generate depth maps and corresponding point clouds.
The resulting point clouds are saved in the specified output directory.

Usage:
    python script.py --encoder vitl --load-from path_to_model --max-depth 20 --img-path path_to_images --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4

Arguments:
    --encoder: Model encoder to use. Choices are ['vits', 'vitb', 'vitl', 'vitg'].
    --load-from: Path to the pre-trained model weights.
    --max-depth: Maximum depth value for the depth map.
    --img-path: Path to the input image or directory containing images.
    --outdir: Directory to save the output point clouds.
    --focal-length-x: Focal length along the x-axis.
    --focal-length-y: Focal length along the y-axis.
"""

import cv2
import numpy as np
import os
from PIL import Image
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from util import kitti_util
from dataset import __datasets__
from torch.utils.data import DataLoader

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from projects.configs.config import CONF

class Depth2Lidar:
    def __init__(self, dataset, model, data_path, test_list, save_dir, sequence):
        self.calib_dir = data_path
        self.data_path = data_path
        self.save_dir = save_dir

        self.sequence = sequence
        self.model = model

        # dataset, dataloader
        KITTIDataset = __datasets__[dataset]
        test_dataset = KITTIDataset(data_path, test_list, False)
        self.TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=8, drop_last=False)

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)


    def test(self):
        for _, sample in enumerate(self.TestImgLoader):
            left_filenames = sample["left_filename"]

            for fn in left_filenames:
                img_path = os.path.join(self.data_path, fn)
                self.process_point_cloud(img_path)

    def process_point_cloud(self, img_path):
        # Process each image file
        # Load the image
        color_image = Image.open(img_path).convert('RGB')
        width, height = color_image.size

        # Read the image using OpenCV
        image = cv2.imread(img_path)
        pred = self.model.infer_image(image, height)

        calib_file = '{}/{}.txt'.format(self.calib_dir, 'calib')
        calib = kitti_util.Calibration(calib_file)

        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        points = np.stack([x, y, pred])
        points = points.reshape((3, -1))
        points = points.T

        cloud = calib.project_image_to_velo(points)

        cloud = np.concatenate([cloud, np.ones((cloud.shape[0], 1))], 1)
        cloud = cloud.astype(np.float32)

        img_name = img_path.split('/')[-1].split('.')[0]

        cloud.tofile('{}/{}.bin'.format(self.save_dir, img_name))

        print('Finish Depth {}/{}'.format(self.sequence, img_name))


def main():
    sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', 
                 '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                 '20', '21']
    
    load_ckpt = os.path.join(CONF.PATH.DEPTHANYTHING, 'ckpt', 'depth_anything_v2_metric_vkitti_vitl.pth')
    max_depth = 80
    dataset = 'kitti'
    encoder = 'vitl'


    # model
    # Determine the device to use (CUDA, MPS, or CPU)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(load_ckpt, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    

    for seq in sequences:
        data_path = os.path.join(CONF.PATH.DATA_DATASETS_SEQUENCES, seq)
        test_list = os.path.join(CONF.PATH.DEPTHANYTHING , 'filenames', (seq + '.txt'))
        save_dir = os.path.join(CONF.PATH.DATA_DATASETS_MSNET3D_LIDAR, 'sequences', seq)

        D2L = Depth2Lidar(
            dataset,
            depth_anything,
            data_path, 
            test_list, 
            save_dir, 
            seq,
        )

        D2L.test()


if __name__ == '__main__':
    main()
