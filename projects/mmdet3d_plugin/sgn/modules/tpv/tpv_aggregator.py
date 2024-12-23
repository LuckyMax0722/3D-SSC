from mmcv.runner import BaseModule
from mmdet.models import HEADS


@HEADS.register_module()
class TPVAggregator(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z,
        scale_h=2, scale_w=2, scale_z=2,
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z

    
    def forward(self, tpv_list):
        """
        tpv_list[0]: bs, c, h*w, c
        tpv_list[1]: bs, c, z*h, c
        tpv_list[2]: bs,cw*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]

        tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
        tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 2, 3).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
        tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1)
    
        fused = tpv_hw + tpv_zh + tpv_wz  # [bs, c, w, h, z]
        fused = fused.permute(0, 1, 3, 2, 4)

        return fused
