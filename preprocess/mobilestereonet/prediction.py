'''
Mathmatical Formula to convert the disparity to depth:

depth = baseline * focal / disparity
For KITTI the baseline is 0.54m and the focal ~721 pixels.
The final formula is:
depth = 0.54 * 721 / disp

For KITTI-360, depth = 0.6 * 552.554261 / disp
'''

from __future__ import print_function, division
import os
import argparse
import torch.nn as nn
from skimage import io
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import __datasets__
from models import __models__
from utils import *
from utils.KittiColormap import *

cudnn.benchmark = True


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from projects.configs.config import CONF

class DepthPrediction:
    def __init__(self, model, maxdisp, dataset, datapath, testlist, loadckpt, colored, num_seq, savepath, baseline):
        self.savepath = savepath
        self.num_seq = num_seq
        self.baseline = baseline
        self.colored = colored

        # dataset, dataloader
        StereoDataset = __datasets__[dataset]
        test_dataset = StereoDataset(datapath, testlist, False)
        self.TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

        # model, optimizer
        self.model = __models__[model](maxdisp)
        self.model = nn.DataParallel(self.model)
        self.model.cuda()

        # load parameters
        print("Loading model {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        self.model.load_state_dict(state_dict['model'])


    def test(self):
        print("Generating the disparity maps...")

        for batch_idx, sample in enumerate(self.TestImgLoader):

            disp_est_tn = self.test_sample(sample)
            disp_est_np = tensor2numpy(disp_est_tn)
            top_pad_np = tensor2numpy(sample["top_pad"])
            right_pad_np = tensor2numpy(sample["right_pad"])
            left_filenames = sample["left_filename"]

            for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):

                assert len(disp_est.shape) == 2

                disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32) 

                # -------------------------------------------------------------------------------------------------------------
                # convert to depth value
                output_folder = os.path.join(self.savepath, "sequences", self.num_seq)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                    
                fn = os.path.join(output_folder, fn.split('/')[-1].split('.')[0])
                depth = self.baseline / disp_est.clip(min=1e-8)
                np.save(fn, depth)

                # depth = 388.1823 / disp_est.clip(min=1e-8) # sequence 0-2; 13-21
                # depth = 381.8293 / disp_est.clip(min=1e-8) # sequence 4-12
                # depth = 389.6304 / disp_est.clip(min=1e-8) # sequence 3
                # depth = 331.532557 / disp_est.clip(min=1e-8) # kitti-360
            
                # -------------------------------------------------------------------------------------------------------------
                # save the disparity image
                output_folder = os.path.join(self.savepath, "disparity", self.num_seq)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                fn = os.path.join(output_folder, fn.split('/')[-1].split('.')[0] + '.jpg')

                print("saving to", fn, disp_est.shape)
                if float(self.colored) == 1:
                    disp_est = kitti_colormap(disp_est)
                    cv2.imwrite(fn, disp_est)
                else:
                    disp_est = np.round(disp_est * 256).astype(np.uint16)
                    io.imsave(fn, disp_est)
                # -------------------------------------------------------------------------------------------------------------
        print("Done!")


    @make_nograd_func
    def test_sample(self, sample):
        self.model.eval()
        disp_ests = self.model(sample['left'].cuda(), sample['right'].cuda())
        return disp_ests[-1]

if __name__ == '__main__':
    
    sequences = {
        '00': 388.1823,
        '01': 388.1823,
        '02': 388.1823,
        '03': 389.6304,
        '04': 381.8293,
        '05': 381.8293,
        '06': 381.8293,
        '07': 381.8293,
        '08': 381.8293,
        '09': 381.8293,
        '10': 381.8293,
        '11': 381.8293,
        '12': 381.8293,
        '13': 388.1823,
        '14': 388.1823,
        '15': 388.1823,
        '16': 388.1823,
        '17': 388.1823,
        '18': 388.1823,
        '19': 388.1823,
        '20': 388.1823,
        '21': 388.1823,
    }

    for id, baseline in sequences.items():
        datapath = os.path.join(CONF.PATH.DATA_DATASETS_SEQUENCES, id)
        testlist = os.path.join(CONF.PATH.MOBILESTEREONET, 'filenames', (id + '.txt'))
        loadckpt = os.path.join(CONF.PATH.MOBILESTEREONET, 'MSNet3D_SF_DS_KITTI2015.ckpt')

        DP = DepthPrediction(
                model = 'MSNet3D',
                maxdisp = 192,
                dataset = 'kitti',
                datapath = datapath,
                testlist = testlist,
                loadckpt = loadckpt,
                colored = 1,
                num_seq = id,
                savepath = CONF.PATH.DATA_DATASETS_MSNET3D_DEPTH,
                baseline = baseline
        )

        DP.test()
        break
