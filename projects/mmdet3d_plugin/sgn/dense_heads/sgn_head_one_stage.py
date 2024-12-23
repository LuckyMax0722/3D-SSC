# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Jianbiao Mei
# ---------------------------------------------

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, builder
from projects.mmdet3d_plugin.sgn.utils.header import Header, SparseHeader
from projects.mmdet3d_plugin.sgn.modules.sgb import SGB
from projects.mmdet3d_plugin.sgn.modules.sdb import SDB
from projects.mmdet3d_plugin.sgn.modules.flosp import FLoSP
from projects.mmdet3d_plugin.sgn.modules.latentnet_v1 import Decoder
from projects.mmdet3d_plugin.sgn.utils.lovasz_losses import lovasz_softmax
from projects.mmdet3d_plugin.sgn.utils.ssc_loss import sem_scal_loss, geo_scal_loss, CE_ssc_loss

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF

@HEADS.register_module()
class SGNHeadOne(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        embed_dims,
        scale_2d_list,
        pts_header_dict,
        latent_header_dict=None,
        depth=3,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag = False,
        **kwargs
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.embed_dims = embed_dims
        self.nvidia_smi = False
        
        if kwargs.get('dataset', 'semantickitti') == 'semantickitti':
            self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                                "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
            self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        elif kwargs.get('dataset', 'semantickitti') == 'kitti360':
            self.class_names =  ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object']
            self.class_weights = torch.from_numpy(np.array([0.464, 0.595, 0.865, 0.871, 0.717, 0.657, 0.852, 0.541, 0.602, 0.567, 0.607, 0.540, 0.636, 0.513, 0.564, 0.701, 0.774, 0.580, 0.690]))
        self.n_classes = len(self.class_names)

        if CONF.LATENTNET.USE_V1:
            self.decoder = Decoder()
        elif CONF.LATENTNET.USE_V4:
            from ..modules.latentnet_v4 import LatentNet
            self.latent = LatentNet()
        elif CONF.LATENTNET.USE_V5 or CONF.LATENTNET.USE_V5_1 or CONF.LATENTNET.USE_V5_2:
            from ..modules.latentnet_v5 import LatentNet
            self.latent = LatentNet()
        elif CONF.LATENTNET.USE_V6 or CONF.LATENTNET.USE_V6_1:
            self.latent = builder.build_head(latent_header_dict)
            
        self.flosp = FLoSP(scale_2d_list)
        
        if CONF.UNCERTAINTY.USE_V1:
            from ..modules.uncertainty_v1 import UncertaintyModel
            self.uncertainty = UncertaintyModel(feature=self.embed_dims, class_num=self.n_classes)
            
        if CONF.FULL_SCALE.USE_V1:
            self.bev_h = 256
            self.bev_w = 256 
            self.bev_z = 32
            
            self.upsampler = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            
            self.embed_dims = self.embed_dims // 2
            
        #self.bottleneck = nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1)
        
        if not CONF.LATENTNET.USE_V6 or not CONF.TPV.USE_V2:
            self.bottleneck = nn.Conv3d(128, self.embed_dims, kernel_size=3, padding=1)
            
        self.sgb = SGB(sizes=[self.bev_h, self.bev_w, self.bev_z], channels=self.embed_dims)
        
        self.mlp_prior = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims//2),
            nn.LayerNorm(self.embed_dims//2),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dims//2, self.embed_dims)
        )
        
        occ_channel = 8 if pts_header_dict.get('guidance', False) else 0
    
        
        if CONF.TPV.USE_V2:
            self.occ_header = nn.Sequential(
                SDB(channel=self.embed_dims, out_channel=20, depth=1),
                nn.Conv3d(20, 1, kernel_size=3, padding=1)
            )
        else:
            self.occ_header = nn.Sequential(
                SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=1),
                nn.Conv3d(self.embed_dims//2, 1, kernel_size=3, padding=1)
            )
            
        self.sem_header = SparseHeader(self.n_classes, feature=self.embed_dims)
        
        self.pts_header = builder.build_head(pts_header_dict)
        
        if CONF.FULL_SCALE.USE_V2:
            from projects.mmdet3d_plugin.sgn.utils.header import HeaderFullScaleV2
            
            c = 20   # origional is 64, too many paras for full scals
            
            self.upsampler = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

            self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=c, depth=depth)
            
            self.ssc_header = HeaderFullScaleV2(self.n_classes, feature=c)
        
        elif CONF.FULL_SCALE.USE_V3:
            from projects.mmdet3d_plugin.sgn.utils.header import HeaderFullScaleV3
            
            self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=self.embed_dims//2, depth=depth)
            self.ssc_header = HeaderFullScaleV3(self.n_classes, feature=self.embed_dims//2)
        
        elif CONF.LATENTNET.USE_V5 or CONF.LATENTNET.USE_V5_1 or CONF.LATENTNET.USE_V5_2: 
            from projects.mmdet3d_plugin.sgn.utils.header import HeaderV5
            
            self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=self.embed_dims//2, depth=depth)
            self.ssc_header = HeaderV5(self.n_classes, feature=self.embed_dims//2)     
            
        elif CONF.LATENTNET.USE_V6:
            self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=40, depth=depth)
            self.ssc_header = Header(self.n_classes, feature=40)   
        
        
        elif CONF.TPV.USE_V2:
            self.combine_coeff = nn.Sequential(
                nn.Conv3d(self.embed_dims, 3, kernel_size=1, bias=False),
                #nn.Softmax(dim=1)
            )
            
            self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=40, depth=depth)
            self.ssc_header = Header(self.n_classes, feature=40)
            
        else:
            self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=self.embed_dims//2, depth=depth)
            self.ssc_header = Header(self.n_classes, feature=self.embed_dims//2)
            
            
            
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.save_flag = save_flag

    def forward(self, mlvl_feats, img_metas, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
                list!!!!
                torch.Size([1, 5, 128, 24, 77])
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
                torch.Size([256, 256, 32])
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """
        out = {}

        if self.nvidia_smi:
            import subprocess
            result = subprocess.run(["nvidia-smi"])
            print(result)
        
        if CONF.LATENTNET.USE_V1:
            mlvl_feats[0] = self.decoder.forward_image(mlvl_feats, img_metas, target)
        
        # View Transformation
        x3d = self.flosp(mlvl_feats, img_metas) # bs, c, nq --> torch.Size([1, 128, 262144])
        bs, c, _ = x3d.shape
        #x3d = x3d.reshape(bs, c, self.bev_h, self.bev_w, self.bev_z)  # torch.Size([1, 128, 128, 128, 16])
        x3d = x3d.reshape(bs, c, 128, 128, 16)  # torch.Size([1, 128, 128, 128, 16])

        if CONF.FULL_SCALE.USE_V1:
            x3d = x3d.permute(0, 1, 4, 3, 2)  # torch.Size([1, 128, 128, 128, 16]) --> torch.Size([1, 128, 16, 128, 128])
            x3d = self.upsampler(x3d)  # torch.Size([1, 128, 128, 128, 16]) --> torch.Size([1, 128, 256, 256, 32])
            x3d = x3d.permute(0, 1, 4, 3, 2)  # torch.Size([1, 128, 256, 256, 32])

        if CONF.LATENTNET.USE_V6:
            if img_metas[0]['mode'] == 'train':
                out['lattent_loss'], x3d = self.latent.forward_train(x3d, target)
            elif img_metas[0]['mode'] == 'test':
                x3d = self.latent.forward_test(x3d, None)

        elif CONF.TPV.USE_V2:
            weights = self.combine_coeff(x3d)
            z = img_metas[0]['latent_feats']
            
            x3d = x3d + z[0] * weights[:, 0:1, ...] + z[1] * weights[:, 1:2, ...] + z[2] * weights[:, 2:3, ...]

            '''
            out_feats:
                torch.Size([1, 128, 128, 128, 16])
            '''
        else:    
            x3d = self.bottleneck(x3d)  # torch.Size([1, 128, 128, 128, 16]) --> torch.Size([1, 128, 128, 128, 16])
    
    
        if CONF.LATENTNET.USE_V4:
            if img_metas[0]['mode'] == 'train':
                x3d, lattent_loss = self.latent.forward_train(x3d, target)
                out['lattent_loss'] = lattent_loss
            elif img_metas[0]['mode'] == 'test':
                x3d = self.latent.forward_test(x3d)

            x3d = self.bottleneck(x3d)  # torch.Size([1, 128, 128, 128, 16]) --> torch.Size([1, 128, 128, 128, 16])
        
        if CONF.LATENTNET.USE_V5:
            if img_metas[0]['mode'] == 'train':
                target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)
                recons_logit, out['recons_loss'], out['vq_loss'], out['lattent_loss'] = self.latent.forward_train(x3d, target_2)

            elif img_metas[0]['mode'] == 'test':
                recons_logit = self.latent.forward_test(x3d)
                
        elif CONF.LATENTNET.USE_V5_1:
            if img_metas[0]['mode'] == 'train':
                target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)
                recons_logit, out['lattent_loss'] = self.latent.forward_train(x3d, target_2)

            elif img_metas[0]['mode'] == 'test':
                recons_logit = self.latent.forward_test(x3d)
        
        elif CONF.LATENTNET.USE_V5_2:
            if img_metas[0]['mode'] == 'train':
                target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)
                recons_logit, out['lattent_loss'] = self.latent.forward_train(x3d, target_2)
            
            elif img_metas[0]['mode'] == 'test':
                recons_logit = self.latent.forward_test(x3d)
            
            recons_logit = recons_logit[:, :20, :, :, :]

        
        # Geometry Guidance --> SDB + 3D Conv
        occ = self.occ_header(x3d).squeeze(1) # ([1, 128, 128, 16])
        out["occ"] = occ

        if CONF.UNCERTAINTY.USE_V1:
            x3d_original = x3d.clone()
        x3d = x3d.reshape(bs, c, -1)  # torch.Size([1, 128, 262144])
        
        # Load proposals
        pts_out = self.pts_header(mlvl_feats, img_metas, target)
        pts_occ = pts_out['occ_logit'].squeeze(1)  # torch.Size([1, 128, 128, 16])

        proposal =  (pts_occ > 0).float().detach().cpu().numpy()  # (1, 128, 128, 16)
        out['pts_occ'] = pts_occ  # torch.Size([1, 128, 128, 16])


        if proposal.sum() < 2:
            proposal = np.ones_like(proposal)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1)>0)).astype(np.int32)  # [[     0      1      2 ... 262132 262136 262142]]
        masked_idx = np.asarray(np.where(proposal.reshape(-1)==0)).astype(np.int32)
        vox_coords = self.get_voxel_indices()  # vox_coords: (262144, 4)

        # vox_coords
        # [[     0      0      0      0]
        # [     0      0      1      1]
        # [     0      0      2      2]
        # ...
        # [   127    127     13 262141]
        # [   127    127     14 262142]
        # [   127    127     15 262143]]

        
        # Compute seed features
        # bs = 1
        # x3d size torch.Size([1, 128, 262144])
        
        seed_feats = x3d[0, :, vox_coords[unmasked_idx[0], 3]].permute(1, 0)
        seed_coords = vox_coords[unmasked_idx[0], :3]
        coords_torch = torch.from_numpy(np.concatenate(
            [np.zeros_like(seed_coords[:, :1]), seed_coords], axis=1)).to(seed_feats.device)
        seed_feats_desc = self.sgb(seed_feats, coords_torch)  # torch.Size([*227962, 128])
        sem = self.sem_header(seed_feats_desc)
        
        out["sem_logit"] = sem  # torch.Size([*227955, 20])
        out["coords"] = seed_coords

        if self.nvidia_smi:
            import subprocess
            result = subprocess.run(["nvidia-smi"])
            print(result)
        
        # Complete voxel features
        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=x3d.device)  # torch.Size([128, 128, 16, 128])
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats_desc
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mlp_prior(x3d[0, :, vox_coords[masked_idx[0], 3]].permute(1, 0))

        vox_feats_diff = vox_feats_flatten.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims).permute(3, 0, 1, 2).unsqueeze(0)  # torch.Size([1, 128, 128, 128, 16])
        
        if self.pts_header.guidance:
            vox_feats_diff = torch.cat([vox_feats_diff, pts_out['occ_x']], dim=1)  # torch.Size([1, 136, 128, 128, 16])  /  torch.Size([1, 136, 256, 256, 32])
        
        if CONF.FULL_SCALE.USE_V2:
            vox_feats_diff = vox_feats_diff.permute(0, 1, 4, 3, 2)  # torch.Size([1, 128, 128, 128, 16]) --> torch.Size([1, 128, 16, 128, 128])
            vox_feats_diff = self.upsampler(vox_feats_diff)  # torch.Size([1, 128, 128, 128, 16]) --> torch.Size([1, 128, 256, 256, 32])
            vox_feats_diff = vox_feats_diff.permute(0, 1, 4, 3, 2)  # torch.Size([1, 128, 256, 256, 32])

        vox_feats_diff = self.sdb(vox_feats_diff) # 1, C,H,W,Z torch.Size([1, 64, 128, 128, 16])  /  torch.Size([1, 32, 256, 256, 32])
    
        if CONF.LATENTNET.USE_V6_1:
            if img_metas[0]['mode'] == 'train':
                out['lattent_loss'], vox_feats_diff = self.latent.forward_train(vox_feats_diff, target)
            elif img_metas[0]['mode'] == 'test':
                vox_feats_diff = self.latent.forward_test(vox_feats_diff, None)
        
        if CONF.LATENTNET.USE_V5 or CONF.LATENTNET.USE_V5_1 or CONF.LATENTNET.USE_V5_2:
            ssc_dict = self.ssc_header(vox_feats_diff, recons_logit)
        else:
            ssc_dict = self.ssc_header(vox_feats_diff)  # --> ssc logit torch.Size([1, 20, 256, 256, 32])
        
        if CONF.UNCERTAINTY.USE_V1:
            ssc_dict = self.uncertainty(x3d_original, ssc_dict)
            
        out.update(ssc_dict)
        
        if self.nvidia_smi:
            import subprocess
            result = subprocess.run(["nvidia-smi"])
            print(result)

        return out

    def step(self, out_dict, target, img_metas, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """
        ssc_pred = out_dict["ssc_logit"]
        
        if step_type== "train":
            sem_pred_2 = out_dict["sem_logit"]

            target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)
            coords = out_dict['coords']
            sp_target_2 = target_2.clone()[0, coords[:, 0], coords[:, 1], coords[:, 2]]
            
            loss_dict = dict()

            class_weight = self.class_weights.type_as(target)
            
            
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
                loss_dict['loss_ssc'] = loss_ssc

            
            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal

            loss_sem = lovasz_softmax(F.softmax(sem_pred_2, dim=1), sp_target_2, ignore=255)
            loss_sem += F.cross_entropy(sem_pred_2, sp_target_2.long(), ignore_index=255)
            loss_dict['loss_sem'] = loss_sem
            
            ones = torch.ones_like(target_2).to(target_2.device)
            target_2_binary = torch.where(torch.logical_or(target_2==255, target_2==0), target_2, ones)

            loss_occ = F.binary_cross_entropy(out_dict['occ'].sigmoid()[target_2_binary!=255], target_2_binary[target_2_binary!=255].float())
            loss_dict['loss_occ'] = loss_occ  # --> L geo

            loss_dict['loss_pts'] = F.binary_cross_entropy(out_dict['pts_occ'].sigmoid()[target_2_binary!=255], target_2_binary[target_2_binary!=255].float())

            if CONF.LATENTNET.USE_V4:
                loss_dict['loss_latent'] = out_dict['lattent_loss']
            
            if CONF.LATENTNET.USE_V5:
                loss_dict['loss_recons'] = out_dict['recons_loss']
                loss_dict['loss_vq'] = out_dict['vq_loss']
                loss_dict['loss_latent'] = out_dict['lattent_loss']
            elif CONF.LATENTNET.USE_V5_1:
                loss_dict['loss_latent'] = out_dict['lattent_loss']
            elif CONF.LATENTNET.USE_V6 or CONF.LATENTNET.USE_V6_1:
                loss_dict['loss_latent'] = out_dict['lattent_loss']
            return loss_dict

        elif step_type== "val" or "test":
            result = dict()
            result['output_voxels'] = ssc_pred
            result['target_voxels'] = target

            if self.save_flag:
                y_pred = ssc_pred.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                self.save_pred(img_metas, y_pred)

            return result

    def training_step(self, out_dict, target, img_metas):
        """Training step.
        """
        return self.step(out_dict, target, img_metas, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, target, img_metas, "val")

    def get_voxel_indices(self):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        return vox_coords

    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        Note: only for semantickitti
        """

        y_pred[y_pred==10] = 44
        y_pred[y_pred==11] = 48
        y_pred[y_pred==12] = 49
        y_pred[y_pred==13] = 50
        y_pred[y_pred==14] = 51
        y_pred[y_pred==15] = 70
        y_pred[y_pred==16] = 71
        y_pred[y_pred==17] = 72
        y_pred[y_pred==18] = 80
        y_pred[y_pred==19] = 81
        y_pred[y_pred==1] = 10
        y_pred[y_pred==2] = 11
        y_pred[y_pred==3] = 15
        y_pred[y_pred==4] = 18
        y_pred[y_pred==5] = 20
        y_pred[y_pred==6] = 30
        y_pred[y_pred==7] = 31
        y_pred[y_pred==8] = 32
        y_pred[y_pred==9] = 40

        # save predictions
        pred_folder = os.path.join("./sgn", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))
