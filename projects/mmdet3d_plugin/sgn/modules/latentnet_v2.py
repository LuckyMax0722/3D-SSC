import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from .pointTransformer.point_transformer import PointsEncoder_pointwise
from .network_img.resnet import Image_ResNet

class LatentNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Hyperparemeters
        
        self.point_transformer = PointsEncoder_pointwise(is_pc_norm = False)
        self.resnet = Image_ResNet()
        
    
    def forward_train(self, img, img_metas, target):
        '''
        input:
            img: torch.Size([1, 5, 3, 370, 1220])
        '''
        device = target.device
        
        pt = torch.from_numpy(img_metas[0]['pc']).unsqueeze(0).contiguous().to(device)
        
        print(img.shape)
        global_pc_feat, point_wised_feat = self.point_transformer(pt) #torch.Size([1, 512]) torch.Size([1, 128, 40960])
        
        print(global_pc_feat.size())
        print(point_wised_feat.size())