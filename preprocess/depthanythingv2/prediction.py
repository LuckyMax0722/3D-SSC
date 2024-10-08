import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from projects.configs.config import CONF

import os
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from dataset import __datasets__
from torch.utils.data import DataLoader

from depth_anything_v2.dpt import DepthAnythingV2
from util.KittiColormap import kitti_colormap

class DepthPrediction:
    def __init__(self, sequence, dataset, datapath, testlist, loadckpt, savepath):
        self.loadckpt = loadckpt
        self.sequence = sequence
        self.savepath = savepath

        # dataset, dataloader
        KITTIDataset = __datasets__[dataset]
        test_dataset = KITTIDataset(datapath, testlist, False)
        self.TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=8, drop_last=False)


    def test(self):
        print("Generating the disparity maps...")

        for batch_idx, sample in enumerate(self.TestImgLoader):
            left_filenames = sample["left_filename"]

            for fn in left_filenames:
                img_path = os.path.join(CONF.PATH.DATA_DATASETS_SEQUENCES, self.sequence, fn)
                self.predict_depth(fn, img_path)
                break
            break

    def predict_depth(self, fn, img_path):     
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        depth_anything = DepthAnythingV2(**{**model_configs['vitl'],  'max_depth': 80}) # can be changed
        depth_anything.load_state_dict(torch.load(self.loadckpt, map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()

        filename = img_path
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, input_size=518)
        print(type(depth))
        # Save Depth Value to .npy file
        output_folder = os.path.join(self.savepath, "sequences", self.sequence)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        fn = os.path.join(output_folder, fn.split('/')[-1].split('.')[0])
        np.save(fn, depth)


        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        # Save Depth Image
        output_folder = os.path.join(self.savepath, "disparity", self.sequence)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        fn = os.path.join(output_folder, fn.split('/')[-1].split('.')[0] + '.jpg')
        
        depth_color = kitti_colormap(depth)
        cv2.imwrite(fn, depth_color)
     
    

if __name__ == '__main__':
    sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', 
                 '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                 '20', '21']
    
    for seq in sequences:
        datapath = os.path.join(CONF.PATH.DATA_DATASETS_SEQUENCES, seq)
        testlist = os.path.join(CONF.PATH.DEPTHANYTHING, 'filenames', (seq + '.txt'))
        loadckpt = os.path.join(CONF.PATH.DEPTHANYTHING, 'ckpt', 'depth_anything_v2_metric_vkitti_vitl.pth')

        DP = DepthPrediction(
                sequence = seq,
                dataset = 'kitti',
                datapath = datapath,
                testlist = testlist,
                loadckpt = loadckpt,
                savepath = CONF.PATH.DATA_DATASETS_MSNET3D_DEPTH,
        )

        DP.test()
        break

