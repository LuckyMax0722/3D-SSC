import torch
import torch.nn as nn
import numpy as np

class Decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, latent_dim):
        super().__init__()
        
        self.spatial_axes = [2, 3]
        
        
        self.Image_decoder_block1 = nn.Sequential(  
            nn.Conv2d(3 + latent_dim, 3, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=1),
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

    def forward_image(self, x, z):
        self.device = x.device
        
        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, x.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, x.shape[self.spatial_axes[1]])
        x = torch.cat((x, z), 1)  # torch.Size([5, 6, 370, 1220])
        
        x = self.Image_decoder_block1(x)

        
        return x.unsqueeze(0) # torch.Size([1, 5, 3, 370, 1220])
