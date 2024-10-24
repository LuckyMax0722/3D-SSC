import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2
from dataset.SemanticKITTI import KITTIDataset 
from torch.utils.data import DataLoader

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from projects.configs.config import CONF

class Image2Depth:
    def __init__(self, model, data_path, test_list, save_dir, sequence):
        self.model = model
        self.save_dir = save_dir
        self.sequence = sequence
        self.input_size = 518
        self.data_path = data_path

        # dataset, dataloader
        test_dataset = KITTIDataset(data_path, test_list, False)
        self.TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)


    def test(self):
        for _, sample in tqdm(enumerate(self.TestImgLoader), total=len(self.TestImgLoader), desc=f"Processing sequence {self.sequence}"):
            left_filenames = sample["left_filename"]
            for fn in left_filenames:
                img_path = os.path.join(self.data_path, fn)
                self.process_depth_image(img_path)


    def process_depth_image(self, filename):         
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, self.input_size)
        
        filename = filename.split('/')[-1].split('.')[0]

        output_path = os.path.join(self.save_dir, filename + '.npy')

        np.save(output_path, depth)
            

if __name__ == '__main__':
    sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', 
                 '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                 '20', '21']
    
    load_from = CONF.PATH.CHECKPOINT_DA
    encoder = 'vitl'
    max_depth = 80

    # model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_anything.load_state_dict(torch.load(load_from, map_location='cpu'))
    depth_anything = depth_anything.to('cuda').eval()

    print("Generating the depth images")

    for seq in sequences:
        data_path = os.path.join(CONF.PATH.DATA_DATASETS_SEQUENCES, seq)
        test_list = os.path.join(CONF.PATH.DEPTHANYTHING , 'filenames', (seq + '.txt'))
        save_dir = os.path.join(CONF.PATH.DATA_DATASETS_MSNET3D_DEPTH , 'sequences', seq)
    
        I2D = Image2Depth(
            depth_anything,
            data_path, 
            test_list, 
            save_dir, 
            seq
        )

        I2D.test()

    
    
    
   