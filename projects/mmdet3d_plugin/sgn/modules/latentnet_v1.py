import numpy as np

import torch
import torch.nn as nn
import torch_scatter
import spconv.pytorch as spconv
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from functools import partial

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF

class MeanVFE(nn.Module):
    def __init__(self, num_point_features):
        super().__init__()
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, voxel_features, voxel_num_points):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        
        return points_mean.contiguous()

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class VoxelBackBone8x(nn.Module):
    def __init__(self, input_channels, grid_size):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]  # process to [z, y, x]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, voxel_features, voxel_coords):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx] wrong! [bs, x, y, z]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        batch_size = 1
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        
        '''
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        '''

        return out

class HeightCompression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoded_spconv_tensor):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """

        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)  # torch.Size([1, 128, 32, 32])

        return spatial_features

class BaseBEVBackbone(nn.Module):
    def __init__(self, input_channels, layer_nums, layer_strides, num_filters, upsample_strides, num_upsample_filters):
        super().__init__()


        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or stride == 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, spatial_features):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """

        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        return spatial_features
    
class Encoder_x(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        
        # Encoder Block x
        # Voxel + Image branch ([1, 352, 32, 32] input)
        channels = 352
        
        self.Prior_encoder_block1 = nn.Sequential(
            nn.Conv2d(int(channels), int(channels*1.25), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(channels*1.25), int(channels*1.25), kernel_size=3, padding=1),
            nn.ReLU(),
        ) #[1, 406, 32, 32]
        
        self.Prior_encoder_block2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(channels*1.25), int(channels*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(channels*1.5), int(channels*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
        ) #[1, 528, 16, 16]
        
        self.Prior_encoder_block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(channels*1.5), int(channels*1.75), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(channels*1.75), int(channels*1.75), kernel_size=3, padding=1),
            nn.ReLU(),
        ) #[1, 616, 8, 8]
        
        self.Prior_encoder_block4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(channels*1.75), int(channels*2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(channels*2), int(channels*2), kernel_size=3, padding=1),
            nn.ReLU(),
        ) #[1, 704, 4, 4]
        
        
        channels_out = 11264
        # [1, 11264]
        self.Prior_encoder_block5 = nn.Sequential(
            nn.Linear(channels_out, 4096), # [1, 4096]
            nn.ReLU(),
            nn.Linear(4096, 1024), # [1, 1024]
            nn.ReLU(),
            nn.Linear(1024, 256), # [1, 256]
            nn.ReLU(),
        )
        
        self.fc1 = nn.Linear(256, latent_size)
        self.fc2 = nn.Linear(256, latent_size)
        
        
    def forward(self, voxel_features, image_features):

        combined_features = torch.cat((voxel_features, image_features), dim=1)  # cat torch.Size([1, 352, 32, 32])

        combined_features = self.Prior_encoder_block1(combined_features)  #[1, 406, 32, 32]
        combined_features = self.Prior_encoder_block2(combined_features)  #[1, 528, 16, 16]
        combined_features = self.Prior_encoder_block3(combined_features)  #[1, 616, 8, 8]
        combined_features = self.Prior_encoder_block4(combined_features)  #[1, 704, 4, 4]
        
        combined_features = combined_features.view(1, -1) # [1, 11264]
        
        combined_features = self.Prior_encoder_block5(combined_features)
        
        mu = self.fc1(combined_features)
        logvar = self.fc2(combined_features)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
      
        return dist, mu, logvar
    
class Encoder_xy(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        
        # Encoder Block x
        # Voxel + Image branch + Target ([1, 352, 32, 32] input)
        channels = 432
        
        self.Prior_encoder_block1 = nn.Sequential(
            nn.Conv2d(int(channels), int(channels*1.25), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(channels*1.25), int(channels*1.25), kernel_size=3, padding=1),
            nn.ReLU(),
        ) #[1, 540, 32, 32]
        
        self.Prior_encoder_block2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(channels*1.25), int(channels*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(channels*1.5), int(channels*1.5), kernel_size=3, padding=1),
            nn.ReLU(),
        ) #[1, 675, 16, 16]
        
        self.Prior_encoder_block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(channels*1.5), int(channels*1.75), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(channels*1.75), int(channels*1.75), kernel_size=3, padding=1),
            nn.ReLU(),
        ) #[1, 843, 8, 8]
        
        self.Prior_encoder_block4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(channels*1.75), int(channels*2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(channels*2), int(channels*2), kernel_size=3, padding=1),
            nn.ReLU(),
        ) #[1, 1054, 4, 4]
        
        
        channels_out = 13824
        # [1, 13824]
        self.Prior_encoder_block5 = nn.Sequential(
            nn.Linear(channels_out, 4096), # [1, 4096]
            nn.ReLU(),
            nn.Linear(4096, 1024), # [1, 1024]
            nn.ReLU(),
            nn.Linear(1024, 256), # [1, 256]
            nn.ReLU(),
        )
        
        self.fc1 = nn.Linear(256, latent_size)
        self.fc2 = nn.Linear(256, latent_size)
        
        
    def forward(self, voxel_features, image_features, target_features):

        combined_features = torch.cat((voxel_features, image_features, target_features), dim=1)  # cat torch.Size([1, 352, 32, 32])

        combined_features = self.Prior_encoder_block1(combined_features)  #[1, 540, 32, 32]
        combined_features = self.Prior_encoder_block2(combined_features)  #[1, 468, 16, 16]
        combined_features = self.Prior_encoder_block3(combined_features)  #[1, 756, 8, 8]
        combined_features = self.Prior_encoder_block4(combined_features)  #[1, 864, 4, 4]

        combined_features = combined_features.view(1, -1) # [1, 11264]

        combined_features = self.Prior_encoder_block5(combined_features)
        
        mu = self.fc1(combined_features)
        logvar = self.fc2(combined_features)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
      
        return dist, mu, logvar
  
class Decoder(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.image_spatial_axes = [3, 4]
        self.point_spatial_axes = [2, 3]
        
        # Image post-process
        image_channel = 256
        
        self.Image_decoder_block1 = nn.Sequential(  
            nn.Conv2d(image_channel, image_channel // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(image_channel // 2, image_channel // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        
        voxel_channel = 160
        
        self.Voxel_decoder_block1 = nn.Sequential(  
            nn.Conv2d(voxel_channel, voxel_channel // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(voxel_channel // 2, voxel_channel // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        
        self.Voxel_decoder_block2 = nn.Sequential(  
            nn.Conv2d(voxel_channel // 2, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
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
    
    def forward_image(self, mlvl_feats, img_metas, target):
        '''
        input:
            mlvl_feats: [list(Torch.tensor)] torch.Size([1, 5, 128, 24, 77])
            target: Torch.tensor torch.Size([1, 256, 256, 32])
        '''
        z = img_metas[0]['z']
        self.device = target.device
        img_feat = mlvl_feats[0]
        
        # post process
        
        # img
        z_img = torch.unsqueeze(z, 2)  # [1, 128, 1]
        z_img = self.tile(z_img, 2, img_feat.shape[self.image_spatial_axes[0]])  # [1, 128, 24]
        
        z_img = torch.unsqueeze(z_img, 3) # [1, 128, 24, 1]
        z_img = self.tile(z_img, 3, img_feat.shape[self.image_spatial_axes[1]])  # [1, 128, 24, 77]
        
        z_img = torch.unsqueeze(z_img, 1)  # [1, 1, 128, 24, 77]
        z_img = z_img.expand(-1, 5, -1, -1, -1) # [1, 5, 128, 24, 77]
        
        img_feat = torch.cat([img_feat, z_img], dim=2)  # [1, 5, 256, 24, 77]
        
        bs, num_cam, c, h, w = img_feat.shape

        img_feat = img_feat.view(bs * num_cam, c, h, w) # [5, 256, 24, 77]
        
        img_feat = self.Image_decoder_block1(img_feat) # [5, 128, 24, 77]
        img_feat = torch.unsqueeze(img_feat, 0)  # [1, 5, 128, 24, 77]
        
        return img_feat

    def forward_point(self, pt_feat, img_metas, target):
        '''
        input:
            pt_feat: (Torch.tensor) torch.Size([1, 32, 256, 256])
            target: Torch.tensor torch.Size([1, 256, 256, 32])
        '''
        z = img_metas[0]['z']
        self.device = target.device

        # post process
        
        # voexl
        z_pt = torch.unsqueeze(z, 2)  # [1, 128, 1]
        z_pt = self.tile(z_pt, 2, pt_feat.shape[self.point_spatial_axes[0]])  # [1, 128, 256]
        
        z_pt = torch.unsqueeze(z_pt, 3) # [1, 128, 256, 256]
        z_pt = self.tile(z_pt, 3, pt_feat.shape[self.point_spatial_axes[1]])  # [1, 128, 256, 256]
        
        pt_feat = torch.cat([pt_feat, z_pt], dim=1)  # [1, 160, 256, 256]
        
        pt_feat = self.Voxel_decoder_block1(pt_feat) # [1, 80, 256, 256]
        pt_feat = self.Voxel_decoder_block2(pt_feat) # [1, 32, 256, 256]
        
        return pt_feat
      
class LatentNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Hyperparemeters
        # VFE backbone3d
        num_point_features = CONF.KITTI.NUM_POINT_FEATURES    
        gird_size = CONF.KITTI.GRID_SIZE
        
        # backbone2d
        input_channels = CONF.PVRCNN.NUM_BEV_FEATURES 
        layer_nums = CONF.PVRCNN.LAYER_NUMS
        layer_strides = CONF.PVRCNN.LAYER_STRIDES
        num_filters = CONF.PVRCNN.NUM_FILTERS
        upsample_strides = CONF.PVRCNN.UPSAMPLE_STRIDES
        num_upsample_filters = CONF.PVRCNN.NUM_UPSAMPLE_FILTERS
        
        # 2. 3 Branches in KL part
        # Point Branch
        self.VFE = MeanVFE(num_point_features)
        gird_size = np.array(gird_size)
        self.backbone3d = VoxelBackBone8x(num_point_features, gird_size)
        self.view_transform = HeightCompression()
        self.backbone2d = BaseBEVBackbone(input_channels, layer_nums, layer_strides, num_filters, upsample_strides, num_upsample_filters)
        
        # Image Branch
        self.Image_encoder_block1 = nn.Sequential(
            nn.Upsample(size=(128, 128), mode='bicubic', align_corners=True),
        )
        
        self.Image_encoder_block2 = nn.Sequential(  
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.Image_encoder_block3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.Image_encoder_block4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=192, out_channels=224, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=224, out_channels=224, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Target Branch
        f = 32
        
        self.Target_encoder_block1 = nn.Sequential(
        nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        self.Target_encoder_block2 = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(f, int(f*1.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(int(f*1.5), int(f*1.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        self.Target_encoder_block3 = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(int(f*1.5), int(f*2), kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(int(f*2), int(f*2), kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )

        self.Target_encoder_block4 = nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(int(f*2), int(f*2.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(int(f*2.5), int(f*2.5), kernel_size=3, padding=1, stride=1),
        nn.ReLU()
        )
        
        
        # Combined Encoder
        latent_size = CONF.LATENTNET.LATENT_DIM
        self.x_encoder = Encoder_x(latent_size)
        self.xy_encoder = Encoder_xy(latent_size)
        
    
    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)
    
    def forward_train(self, mlvl_feats, img_metas, target):
        '''
        input:
            mlvl_feats: [list(Torch.tensor)] torch.Size([1, 5, 128, 24, 77])
            target: Torch.tensor torch.Size([1, 256, 256, 32])
        '''
        
        
        device = target.device

        for i, img_meta in enumerate(img_metas):
            lidar_voxels = torch.from_numpy(img_meta['lidar_voxels']).float().to(device)
            lidar_coordinates = torch.from_numpy(img_meta['lidar_coordinates']).to(device)
            lidar_num_points = torch.from_numpy(img_meta['lidar_num_points']).to(device)

        # Lidar_Point_Voxel Branch
        lidar_voxel_features = self.VFE(lidar_voxels, lidar_num_points)  # torch.Size([109869, 4])
        
        spatial_features = self.backbone3d(lidar_voxel_features, lidar_coordinates)
        spatial_features = self.view_transform(spatial_features)  # torch.Size([1, 128, 32, 32])
        spatial_features = self.backbone2d(spatial_features) # torch.Size([1, 128, 32, 32])
        
        # Image Branch
        image_features = torch.mean(mlvl_feats[0], dim=1)  # flatten [1, 128, 24, 77]
        
        image_features = self.Image_encoder_block1(image_features)
        image_features = self.Image_encoder_block2(image_features)
        image_features = self.Image_encoder_block3(image_features)
        image_features = self.Image_encoder_block4(image_features) # torch.Size([1, 224, 32, 32])
        
        # Target Branch
        target_features = target.permute(0, 3, 2, 1)
        target_features = self.Target_encoder_block1(target_features)
        target_features = self.Target_encoder_block2(target_features)
        target_features = self.Target_encoder_block3(target_features)
        target_features = self.Target_encoder_block4(target_features) # torch.Size([1, 80, 32, 32])
        

        # encoder_x and encoder_xy
        self.prior, mux, logvarx = self.x_encoder(spatial_features, image_features)  # torch.Size [1, 352, 32, 32]
        self.posterior, muxy, logvarxy = self.xy_encoder(spatial_features, image_features, target_features)
        
        z_noise_post = self.reparametrize(muxy, logvarxy) # [1, 128]
        
        img_metas[0]['z'] = z_noise_post
        
        loss_dict = dict()
        lattent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))
        loss_dict['loss_latent'] = lattent_loss
        
        return loss_dict
    
    def forward_test(self, mlvl_feats, img_metas, target):
        '''
        input:
            mlvl_feats: [list(Torch.tensor)] torch.Size([1, 5, 128, 24, 77])
            target: Torch.tensor torch.Size([1, 256, 256, 32])
        '''
        
        device = target.device

        for i, img_meta in enumerate(img_metas):
            lidar_voxels = torch.from_numpy(img_meta['lidar_voxels']).float().to(device)
            lidar_coordinates = torch.from_numpy(img_meta['lidar_coordinates']).to(device)
            lidar_num_points = torch.from_numpy(img_meta['lidar_num_points']).to(device)

        # Lidar_Point_Voxel Branch
        lidar_voxel_features = self.VFE(lidar_voxels, lidar_num_points)  # torch.Size([109869, 4])
        
        spatial_features = self.backbone3d(lidar_voxel_features, lidar_coordinates)
        spatial_features = self.view_transform(spatial_features)  # torch.Size([1, 128, 32, 32])
        spatial_features = self.backbone2d(spatial_features) # torch.Size([1, 128, 32, 32])
        
        # Image Branch
        image_features = torch.mean(mlvl_feats[0], dim=1)  # flatten [1, 128, 24, 77]
        
        image_features = self.Image_encoder_block1(image_features)
        image_features = self.Image_encoder_block2(image_features)
        image_features = self.Image_encoder_block3(image_features)
        image_features = self.Image_encoder_block4(image_features) # torch.Size([1, 224, 32, 32])
        

        # encoder_x and encoder_xy
        self.prior, mux, logvarx = self.x_encoder(spatial_features, image_features)  # torch.Size [1, 352, 32, 32]
        
        z_noise_prior = self.reparametrize(mux, logvarx)  # [1, 128]
        
        img_metas[0]['z'] = z_noise_prior
        

        
        

        
        