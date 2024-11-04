import torch.nn as nn

                                                        
class Header(nn.Module):
    def __init__(
        self,
        class_num,
        feature,
    ):
        super(Header, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )

        self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)


    def forward(self, x3d_l1):
        # [1, 64, 128, 128, 16]
        res = {} 

        x3d_up_l1 = self.up_scale_2(x3d_l1) # [1, dim, 128, 128, 16] -> [1, dim, 256, 256, 32]

        _, feat_dim, w, l, h  = x3d_up_l1.shape

        x3d_up_l1 = x3d_up_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_up_l1)

        ssc_logit = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)
        
        res["ssc_logit"] = ssc_logit

        return res

class Header_1(nn.Module):
    def __init__(
        self,
        class_num,
        feature,
    ):
        super(Header_1, self).__init__()
        self.feature = feature
        self.class_num = class_num
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature),
            nn.Linear(self.feature, self.class_num),
        )


    def forward(self, x3d_l1):
        # [1, 64, 256, 256, 32]
        res = {} 

        _, feat_dim, w, l, h  = x3d_l1.shape

        x3d_l1 = x3d_l1.squeeze().permute(1,2,3,0).reshape(-1, feat_dim)

        ssc_logit_full = self.mlp_head(x3d_l1)

        ssc_logit = ssc_logit_full.reshape(w, l, h, self.class_num).permute(3,0,1,2).unsqueeze(0)
        
        res["ssc_logit"] = ssc_logit

        return res
    
class SparseHeader(nn.Module):
    def __init__(self, class_num, feature):
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature),
            nn.Linear(feature, class_num)
        )

    def forward(self, x):
        x = self.mlp_head(x)

        return x

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          