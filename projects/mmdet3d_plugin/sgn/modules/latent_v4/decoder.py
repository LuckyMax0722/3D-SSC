import torch
import torch.nn as nn
import numpy as np

class Decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, latent_dim):
        super().__init__()
        
        self.spatial_axes = [2, 3, 4]
         
        voxel_channel = 256
        
        self.Voxel_encoder_block1 = nn.Sequential(
            nn.Conv3d(int(voxel_channel), int(voxel_channel // 2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(int(voxel_channel // 2), int(voxel_channel // 2), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(self.device)
        return torch.index_select(a, dim, order_index)

    def forward_voxel(self, x, z):
        self.device = x.device
        
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, x.shape[self.spatial_axes[0]])  # torch.Size([1, 128, 128])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, x.shape[self.spatial_axes[1]])  # torch.Size([1, 128, 128])
        z = torch.unsqueeze(z, 4)
        z = self.tile(z, 4, x.shape[self.spatial_axes[2]])  # torch.Size([1, 128, 128, 32])
        
        x = torch.cat((x, z), 1)  # torch.Size([1, 256, 128, 128, 16])
        
        x = self.Voxel_encoder_block1(x)
        
        return x  # torch.Size([1, 128, 128, 128, 16])
