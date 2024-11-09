import torch
import torch.nn as nn

class UncertaintyModel(nn.Module):
    def __init__(self, feature, class_num):
        super(UncertaintyModel, self).__init__()
        
        self.feature = feature
        self.class_num = class_num
        self.uncertainty = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        
    def forward(self, voxel_features, ssc_dict):
        
        voxel_features_2 = self.up_scale_2(voxel_features) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]
        
        _, feat_dim, w, l, h  = voxel_features_2.shape
        
        voxel_features_2 = voxel_features_2.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)  #  torch.Size([2097152, 128])
        
        voxel_features_2 = self.uncertainty(voxel_features_2)  #  torch.Size([2097152, 20])
        
        voxel_features_2 = torch.exp(voxel_features_2)  # sigma for each voxel
        
        voxel_features_2 = voxel_features_2.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)  # torch.Size([1, 20, 256, 256, 32])
        
        ssc_logit = ssc_dict["ssc_logit"]
        
        # Sample epsilon from a normal distribution
        epsilon = torch.randn_like(ssc_logit)
        
        # Scale noise by uncertainty and add to logits
        ssc_logit = ssc_logit + epsilon * voxel_features_2

        ssc_dict["ssc_logit"] = ssc_logit
        
        return ssc_dict
    