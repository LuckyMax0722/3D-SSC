import os
from os import path as osp
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC

import open3d as o3d

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF


tv = None
try:
    import cumm.tensorview as tv
except:
    pass

@DATASETS.register_module()
class SemanticKittiDataset(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        preprocess_root,
        img_size=[370, 1220],
        temporal = [],
        eval_range = 51.2,
        depthmodel="msnet3d",
        labels_tag = 'labels',
        color_jitter=None,
        scale=2
    ):
        super().__init__()
        
        # KL data preprocess
        if CONF.LATENTNET.USE_V1:
            point_cloud_range = CONF.KITTI.POINT_CLOUD_RANGE
            encoding_type = CONF.KITTI.POINT_ENCODING_TYPE
            feature_list = CONF.KITTI.POINT_FEATURE_LIST
            voxel_size = CONF.KITTI.VOXEL_SIZE
            
            self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
            
            self.point_feature_encoder = PointFeatureEncoder(
                encoding_type,
                feature_list,
                point_cloud_range=self.point_cloud_range
            )
            
            self.data_processor = DataProcessorV1(
                point_cloud_range = point_cloud_range, 
                num_point_features=self.point_feature_encoder.num_point_features, 
                voxel_size = voxel_size
            )
        
        elif CONF.LATENTNET.USE_V2:
            point_cloud_range = CONF.KITTI.POINT_CLOUD_RANGE
            voxel_size = CONF.KITTI.VOXEL_SIZE
            input_pt_num = CONF.VP2P.INPUT_PT_NUM
            
            self.data_processor = DataProcessorV2(
                point_cloud_range = point_cloud_range, 
                voxel_size = voxel_size,
                input_pt_num = input_pt_num,
            )
        
        self.data_root = data_root
        self.label_root = os.path.join(preprocess_root, labels_tag)
        self.depth_query = "msnet3d"
        self.depthmodel = depthmodel
        self.eval_range = eval_range
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.split = split 
        self.sequences = splits[split]
        
        self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", 
                            "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                            "parking", "sidewalk", "other-ground", "building", "fence", 
                            "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
        
        self.scene_size = (51.2, 51.2, 6.4)
        self.vox_origin = np.array([0, -25.6, -2])
        self.voxel_size = 0.2  # 0.2m
        self.scale = scale

        self.img_W = img_size[1]
        self.img_H = img_size[0]

        self.poses=self.load_poses()
        self.target_frames = temporal
        self.load_scans()
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_mode = test_mode
        self.set_group_flag()
        

    def __getitem__(self, index):
        
        return self.prepare_data(index)

    def __len__(self):
        return len(self.scans)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def load_poses(self):
        """ read poses for each sequence

            Returns
            -------
            dict
                pose dict for different sequences.
        """
        pose_dict = dict()
        for sequence in self.sequences:
            pose_path = os.path.join(self.data_root, "dataset", "sequences", sequence, "poses.txt")
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            pose_dict[sequence] = self.parse_poses(pose_path, calib)
        return pose_dict

    def load_scans(self):
        """ read each scan

            Returns
            -------
            list
                list of each single scan.
        """
        self.scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "voxels", "*.bin"
            )

            for voxel_path in glob.glob(glob_path):

                self.scans.append(
                    {
                        "sequence": sequence,
                        "pose": self.poses[sequence],
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path
                    }
                )

    def set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def prepare_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []
        example = self.get_data_info(index)

        data_queue.insert(0, example)

        return self.union2one(data_queue)

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'] for each in queue]
        metas_map = {}
        
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """
        scan = self.scans[index]

        voxel_path = scan["voxel_path"]

        sequence = scan["sequence"]
        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        meta_dict = self.get_meta_info(scan, sequence, frame_id)
        img = self.get_input_info(sequence, frame_id)
        target = self.get_gt_info(sequence, frame_id)

        data_info = dict(
            img_metas = meta_dict,
            img = img,
            target = target
        )
        return data_info

    def vox2pix(self, cam_E, cam_k, img_W=None, img_H=None, voxel_size=None, vox_origin=None, scene_size=None):
        """
        compute the 2D projection of voxels centroids
        
        Parameters:
        ----------
        cam_E: 4x4
        =camera pose in case of NYUv2 dataset
        =Transformation from camera to lidar coordinate in case of SemKITTI
        cam_k: 3x3
            camera intrinsics
        vox_origin: (3,)
            lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
        img_W: int
            image width
        img_H: int
            image height
        scene_size: (3,)
            scene size in meter: (51.2, 51.2, 6.4) for SemKITTI
        
        Returns
        -------
        projected_pix: (N, 2)
            Projected 2D positions of voxels
        fov_mask: (N,)
            Voxels mask indice voxels inside image's FOV 
        pix_z: (N,)
            Voxels'distance to the sensor in meter
        """
        img_W, img_H, voxel_size, vox_origin, scene_size = self.img_W, self.img_H, \
            self.scale*self.voxel_size, self.vox_origin, self.scene_size
        # Compute the x, y, z bounding of the scene in meter
        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels centroids in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        xv, yv, zv = np.meshgrid(
                range(vol_dim[0]),
                range(vol_dim[1]),
                range(vol_dim[2]),
                indexing='ij'
            )
        vox_coords = np.concatenate([
                xv.reshape(1,-1),
                yv.reshape(1,-1),
                zv.reshape(1,-1)
            ], axis=0).astype(int).T

        # Project voxels'centroid from lidar coordinates to camera coordinates
        cam_pts = vox2world(vox_origin, vox_coords, voxel_size)
        cam_pts = rigid_transform(cam_pts, cam_E)

        # Project camera coordinates to pixel positions
        projected_pix = cam2pix(cam_pts, cam_k)
        pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

        # Eliminate pixels outside view frustum
        pix_z = cam_pts[:, 2]
        fov_mask = np.logical_and(pix_x >= 0,
                    np.logical_and(pix_x < img_W,
                    np.logical_and(pix_y >= 0,
                    np.logical_and(pix_y < img_H,
                    pix_z > 0))))


        return projected_pix, fov_mask

    def get_meta_info(self, scan, sequence, frame_id):
        """Get meta info according to the given index.

        Args:
            scan (dict): scan information,
            sequence (str): sequence id,
            frame_id (str): frame id

        Returns:
            dict: Meta information that will be passed to the data \
                preprocessing pipelines.
        """
        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )

        # for multiple images
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        image_paths = []

        projected_pixs = []
        fov_masks = []

        pts_list = []

        # transform points from lidar to camera coordinate
        lidar2cam_rt = scan["T_velo_2_cam"]
        # camera intrisic
        P = scan["P"]
        cam_k = P[0:3, 0:3]
        intrinsic = cam_k
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        # transform 3d point in lidar coordinate to 2D image (projection matrix)
        lidar2img_rt = (viewpad @ lidar2cam_rt)

        pts_filename = os.path.join(
                self.data_root, "dataset", f"sequences_{self.depthmodel}_lidar/sequences", sequence, frame_id + ".bin"
            )
        pts = np.fromfile(pts_filename, dtype=np.float32)
        pts = pts.reshape((-1, 4))
        pts = pts[:, :3]

        lidar2img_rts.append(lidar2img_rt)
        lidar2cam_rts.append(lidar2cam_rt)
        cam_intrinsics.append(intrinsic)
        image_paths.append(rgb_path)

        pix, mask = self.vox2pix(lidar2cam_rt, intrinsic)
        projected_pixs.append(pix)
        fov_masks.append(mask)

        pts_list.append(pts)

        # for reference img
        seq_len = len(self.poses[sequence])
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )

            pose_list = self.poses[sequence]

            ref = pose_list[int(frame_id)] # reference frame with GT semantic voxel
            target = pose_list[int(target_id)]
            ref2target = np.matmul(inv(target), ref) # both for lidar

            target2cam = scan["T_velo_2_cam"] # lidar to camera
            ref2cam = target2cam @ ref2target

            lidar2cam_rt  = ref2cam
            lidar2img_rt = (viewpad @ lidar2cam_rt)

            pts_filename = os.path.join(
                self.data_root, "dataset", f"sequences_{self.depthmodel}_lidar/sequences", sequence, target_id + ".bin"
            )
            pts = np.fromfile(pts_filename, dtype=np.float32)
            pts = pts.reshape((-1, 4))
            pts = np.matmul(np.matmul(inv(ref), target), pts.T).T
            pts = pts[:, :3]  # (N, 3)

            lidar2img_rts.append(lidar2img_rt)
            lidar2cam_rts.append(lidar2cam_rt)
            cam_intrinsics.append(intrinsic)
            image_paths.append(rgb_path)

            pix, mask = self.vox2pix(lidar2cam_rt, intrinsic)
            projected_pixs.append(pix)
            fov_masks.append(mask)

            pts_list.append(pts)

        pts_list = np.concatenate(pts_list)
        
        
        if CONF.LATENTNET.USE_V1:   
            # Lidar_Point_Voxel process
            zeros_column = np.zeros((pts_list.shape[0], 1))
            pts_list_4 = np.hstack((pts_list, zeros_column))

            points, use_lead_xyz = self.point_feature_encoder.forward(pts_list_4)

            points = self.data_processor.mask_points(points)
        
            if self.split == 'train' or self.split == 'val':
                points = self.data_processor.shuffle_points(points)

            voxels, coordinates, num_points = self.data_processor.transform_points_to_voxels(points, use_lead_xyz)
            

            #coordinates[:, [0, 2]] = coordinates[:, [2, 0]] # [z, y, x] --> [x, y, z]

            # coordinates [n, 3] but backbone3d need [n, 4] --> [n, bs + 3]
            n = coordinates.shape[0]

            # add 0 as batch size
            coordinates = np.hstack((np.zeros((n, 1)), coordinates))
            
        elif CONF.LATENTNET.USE_V2:

            pc_np = self.data_processor.mask_points(pts_list)
        
            if self.split == 'train' or self.split == 'val':
                pc_np = self.data_processor.shuffle_points(pc_np)
            
            pc_np = self.data_processor.downsample(pc_np.T)
            
            pc_np = self.data_processor.downsample_np(pc_np)  # [3, 40960]
            
        elif CONF.LATENTNET.USE_V3:
            seq_len = len(self.poses[sequence])
            depth_list = []

            depth_path = os.path.join(
                CONF.PATH.DATA_DATASETS_MSNET3D_DEPTH, "sequences", sequence, frame_id + ".npy"
            )
            
            depth = np.load(depth_path)
            
            depth = depth[:self.img_H, :self.img_W]  # crop depth [H, W]
            
            depth = np.clip(depth, 0, 80) # max depth = 80
            depth = depth / 80.0  # normalize
            
            depth = np.expand_dims(depth, axis=(0)) # [1, 370, 1220] [C, H, W]
            
            depth = torch.from_numpy(depth).float()
            
            depth_list.append(depth)
            
            # reference frame
            for i in self.target_frames:
                id = int(frame_id)

                if id + i < 0 or id + i > seq_len-1:
                    target_id = frame_id
                else:
                    target_id = str(id + i).zfill(6)
                
                depth_path = os.path.join(
                    CONF.PATH.DATA_DATASETS_MSNET3D_DEPTH, "sequences", sequence, target_id + ".npy"
                )
                
                depth = np.load(depth_path)
                
                depth = depth[:self.img_H, :self.img_W]  # crop depth [H, W]
            
                depth = np.clip(depth, 0, 80) # max depth = 80
                depth = depth / 80.0  # normalize
                
                depth = np.expand_dims(depth, axis=(0)) # [1, 370, 1220] [C, H, W]
                
                depth = torch.from_numpy(depth).float()
            
                depth_list.append(depth)

            depth_tensor = torch.stack(depth_list, dim=0) #[5, 1, 370, 1220]
         
        else:
            pass
            
        # load ground truth
        if self.split == 'train' or self.split == 'val':
            target_1_2_path = os.path.join(self.label_root, sequence, frame_id + "_1_2.npy")
            target_1_2 = np.load(target_1_2_path)
            target_1_2 = target_1_2.reshape(-1)
            target_1_2 = target_1_2.reshape(128, 128, 16)
            target_1_2 = target_1_2.astype(np.float32)
        else:
            target_1_2 = None

        
        # original
        meta_dict = dict(
                sequence_id = sequence,
                frame_id = frame_id,
                lidar=pts_list,
                target_1_2=target_1_2,
                projected_pix=projected_pixs,
                fov_mask=fov_masks, 
                img_filename=image_paths,
                lidar2img = lidar2img_rts,
                lidar2cam=lidar2cam_rts,
                cam_intrinsic=cam_intrinsics,
                img_shape = [(self.img_H,self.img_W)]
        )
        
        if CONF.LATENTNET.USE_V1:
            meta_dict['lidar_voxels'] = voxels
            meta_dict['lidar_coordinates'] = coordinates
            meta_dict['lidar_num_points'] = num_points
        elif CONF.LATENTNET.USE_V2:
            meta_dict['pc'] = pc_np.astype(np.float32)
        elif CONF.LATENTNET.USE_V3:
            meta_dict['depth_tensor'] = depth_tensor
        else:
            pass
        
        return meta_dict

    def get_input_info(self, sequence, frame_id):
        """Get the image of the specific frame in a sequence.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            torch.tensor: Img.
        """
        
        #  original
        
        seq_len = len(self.poses[sequence])
        image_list = []

        rgb_path = os.path.join(
            self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
        )
        
        img = Image.open(rgb_path).convert("RGB")
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0
        img = img[:self.img_H, :self.img_W, :]  # crop image
        img = self.normalize_rgb(img)
        
        image_list.append(img)

        # reference frame
        for i in self.target_frames:
            id = int(frame_id)

            if id + i < 0 or id + i > seq_len-1:
                target_id = frame_id
            else:
                target_id = str(id + i).zfill(6)

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
            )
            
            img = Image.open(rgb_path).convert("RGB")
            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            img = img[:self.img_H, :self.img_W, :]  # crop image

            img = self.normalize_rgb(img)

            image_list.append(img)

        image_tensor = torch.stack(image_list, dim=0) #[N, 3, 370, 1220]
         
        return image_tensor
        
        
        
        ##############################################
        # Warning!!!!!!!!! Still not finish
        ##############################################
        
        
        
        if CONF.LATENTNET.USE_V2:
            seq_len = len(self.poses[sequence])
            image_list = []

            rgb_path = os.path.join(
                self.data_root, "dataset", "sequences", sequence, "image_2", frame_id + ".png"
            )
            img = Image.open(rgb_path).convert("RGB")
            # Image augmentation
            if self.color_jitter is not None:
                img = self.color_jitter(img)
            # PIL to numpy
            img = np.array(img, dtype=np.float32, copy=False) / 255.0
            
            import cv2
            
            img = cv2.resize(img, (512, 160), interpolation=cv2.INTER_LINEAR)
            
            image_list.append(self.normalize_rgb(img))

            # reference frame
            for i in self.target_frames:
                id = int(frame_id)

                if id + i < 0 or id + i > seq_len-1:
                    target_id = frame_id
                else:
                    target_id = str(id + i).zfill(6)

                rgb_path = os.path.join(
                    self.data_root, "dataset", "sequences", sequence, "image_2", target_id + ".png"
                )
                img = Image.open(rgb_path).convert("RGB")
                # Image augmentation
                if self.color_jitter is not None:
                    img = self.color_jitter(img)
                # PIL to numpy
                img = np.array(img, dtype=np.float32, copy=False) / 255.0

                img = cv2.resize(img, (512, 160), interpolation=cv2.INTER_LINEAR)
                
                img = self.normalize_rgb(img)
                
                image_list.append(img)

            image_tensor = torch.stack(image_list, dim=0) #[5, 3, 160, 512]

            return image_tensor
            

            

    def get_gt_info(self, sequence, frame_id):
        """Get the ground truth.

        Args:
            sequence (str): sequence id,
            frame_id (str): frame id.

        Returns:
            array: target. 
        """
        if self.split == "train" or self.split == "val":
            # load full-range groundtruth
            target_1_path = os.path.join(self.label_root, sequence, frame_id + "_1_1.npy")
            target = np.load(target_1_path)
            # short-range groundtruth
            if self.eval_range == 25.6:
                target[128:, :, :] = 255
                target[:, :64, :] = 255
                target[:, 192:, :] = 255

            elif self.eval_range == 12.8:
                target[64:, :, :] = 255
                target[:, :96, :] = 255
                target[:, 160:, :] = 255
        else:
            target = np.ones((256,256,32))

        return target

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='ssc',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in SemanticKITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        if results is None:
            logger.info('Skip Evaluation')

        if 'ssc_scores' in results:
            # for single-GPU inference
            ssc_scores = results['ssc_scores']
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            # for multi-GPU inference
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])

            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])

            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)

            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }

        for name, iou in zip(self.class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou

        eval_results = {}
        for key, val in res_dic.items():
            eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)

        eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']

        if logger is not None:
            logger.info('SemanticKITTI SSC Evaluation')
            logger.info(eval_results)

        return eval_results

class PointFeatureEncoder(object):
    def __init__(self, encoding_type, feature_list, point_cloud_range=None):
        super().__init__()
        assert list(feature_list[0:3]) == ['x', 'y', 'z']
        self.encoding_type = encoding_type
        self.feature_list = feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        return getattr(self, self.encoding_type)(points=None)

    def forward(self, points):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        points, use_lead_xyz = getattr(self, self.encoding_type)(points)
        
        return points, use_lead_xyz

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.feature_list)
            return num_output_features

        assert points.shape[-1] == len(self.feature_list)
        point_feature_list = [points[:, 0:3]]
        for x in self.feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        
        return point_features, True

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points
     
class DataProcessorV1(object):
    def __init__(self, point_cloud_range, num_point_features, voxel_size):
        self.point_cloud_range = point_cloud_range

        self.num_point_features = num_point_features
        
        self.voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.num_point_features,
            max_num_points_per_voxel=10,
            max_num_voxels=500000,
        )

    def mask_points_by_range(self, points, limit_range):
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
            & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
        return mask

    def mask_points(self, points):
        mask = self.mask_points_by_range(points, self.point_cloud_range)
        
        return points[mask]
            
    def shuffle_points(self, points):
        shuffle_idx = np.random.permutation(points.shape[0])
        points = points[shuffle_idx]

        return points

    def transform_points_to_voxels(self, points, use_lead_xyz):
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not use_lead_xyz:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)
            
        return voxels, coordinates, num_points

class DataProcessorV2:
    def __init__(self, point_cloud_range, voxel_size, input_pt_num):
        self.point_cloud_range = point_cloud_range
        self.voxel_grid_downsample_size = voxel_size[0]
        self.num_pc = input_pt_num
        
    def mask_points_by_range(self, points, limit_range):
        mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
            & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
        return mask

    def mask_points(self, points):
        mask = self.mask_points_by_range(points, self.point_cloud_range)
        
        return points[mask]
            
    def shuffle_points(self, points):
        shuffle_idx = np.random.permutation(points.shape[0])
        points = points[shuffle_idx]

        return points
    
    def downsample(self, pointcloud):
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(np.transpose(pointcloud))

        fake_colors=np.zeros((pointcloud.shape[1],3))

        pcd.colors=o3d.utility.Vector3dVector(fake_colors)

        down_pcd=pcd.voxel_down_sample(voxel_size=self.voxel_grid_downsample_size)
        down_pcd_points=np.transpose(np.asarray(down_pcd.points))

        return down_pcd_points
    
    def downsample_np(self, pc_np):
        if pc_np.shape[1] >= self.num_pc:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pc, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pc:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pc - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        
        return pc_np
    
def vox2world(vol_origin, vox_coords, vox_size, offsets=(0.5, 0.5, 0.5)):
        """Convert voxel grid coordinates to world coordinates."""
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        offsets = np.array(offsets).astype(np.float32)
        cam_pts = vol_origin[None] + vox_size*(vox_coords + offsets[None])

        return cam_pts


def cam2pix(cam_pts, intr):
    """Convert camera coordinates to pixel coordinates."""
    intr = intr.astype(np.float32)
    pix = np.dot(intr, cam_pts.T).T
    pix = pix[:, :2] / np.maximum(pix[:, 2:3], np.ones_like(pix[:, 2:3])*1e-6)

    return np.round(pix).astype(np.int64)

def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]
    
def get_pointcloud(self, pc_folder, seq_i):
    pc_path = os.path.join(pc_folder, '%06d.npy' % seq_i)
    npy_data = np.load(pc_path).astype(np.float32)
    # shuffle the point cloud data, this is necessary!
    npy_data = npy_data[:, np.random.permutation(npy_data.shape[1])]
    pc_np = npy_data[0:3, :]  # 3xN
    intensity_np = npy_data[3:4, :]  # 1xN
    sn_np = npy_data[4:7, :]  # 3xN

    return pc_np, intensity_np, sn_np
    