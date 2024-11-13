import torch
from torch import nn

from .unet3d import UNet3D

class mlp_unet(torch.nn.Module):
    def __init__(self, channel, out_channel, feature, class_num):
        super(mlp_unet, self).__init__()
        self.class_num = class_num
        
        self.conv_in = nn.Conv3d(channel, out_channel, kernel_size=3, padding=1)  # torch.Size([1, 64, 128, 128, 16])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature),
            nn.Linear(feature, class_num),
        )
        
        self.unet = UNet3D(in_channels=class_num, out_channels=class_num)

    def upscale(self, x):
        x = self.conv_in(x)  # torch.Size([1, 128, 128, 128, 16]) --> torch.Size([1, 64, 128, 128, 16]) 

        return x
    
    def sem_head(self, x):
        _, feat_dim, w, l, h  = x.shape

        x = x.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)  # --> torch.Size([262,144, 64])

        x = self.mlp_head(x)  # --> torch.Size([262,144, 20])

        x = x.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)  # --> torch.Size([1, 20, 128, 128, 16])
        
        return x
    
    def encode(self, x):
        x = self.unet.forward_encoder(x)
        
        return x

    def forward(self, x):
        x = self.upscale(x)
        
        x = self.sem_head(x)
        
        x = self.encode(x)

        
        return x


        