import os
import glob
import numpy as np
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet.datasets.pipelines import Compose

from mmdet.datasets import build_dataset

@DATASETS.register_module()
class SemanticKITTIDataset(Dataset):
    def __init__(
        self,
        data_root,
        stereo_depth_root,
        ann_file,
        pipeline,
        split,
        camera_used,
        occ_size,
        pc_range,
        test_mode=False,
        load_continuous=False
    ):
        super().__init__()

        self.load_continuous = load_continuous
        self.splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["08"],
            "test_submit": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }

        self.sequences = self.splits[split]

        self.data_root = data_root
        self.stereo_depth_root = stereo_depth_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.data_infos = self.load_annotations(self.ann_file)

        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'left': '2', 'right': '3'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def __len__(self):
        return len(self.data_infos)
    
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        example = self.pipeline(input_dict)
        return example
    
    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        example = self.pipeline(input_dict)
        return example
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data
    
    def get_data_info(self, index):
        info = self.data_infos[index]
        '''
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
        '''

        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            sequence = info['sequence'],
            frame_id = info['frame_id'],
        )

        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []

        for cam_type in self.camera_used:
            image_paths.append(info['img_{}_path'.format(int(cam_type))])
            lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])
            cam_intrinsics.append(info['P{}'.format(int(cam_type))])
            lidar2cam_rts.append(info['T_velo_2_cam'])
        
        focal_length = info['P2'][0, 0]
        baseline = self.dynamic_baseline(info)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                focal_length=focal_length,
                baseline=baseline
            ))
        input_dict['stereo_depth_path'] = info['stereo_depth_path']
        # gt_occ is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index, key='voxel_path')
        input_dict['gt_occ_1_2'] = self.get_ann_info(index, key='voxel_1_2_path')

        return input_dict
    
    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "sequences", sequence, "calib.txt")
            )
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, "sequences", sequence)
                        
            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, "sequences", sequence, 'image_2', '*.png')
            else:
                id_base_path = os.path.join(self.data_root, "sequences", sequence, 'voxels', '*.bin')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                img_3_path = os.path.join(img_base_path, 'image_3', img_id + '.png')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')
                voxel_1_2_path = os.path.join(voxel_base_path, img_id + '_1_2.npy')
                stereo_depth_path = os.path.join(self.stereo_depth_root, "sequences", sequence, img_id + '.npy')
                
                # for sweep demo or test submission
                if not os.path.exists(voxel_path):
                    voxel_path = None
                
                
                scans.append(
                    {   "img_2_path": img_2_path,
                        "img_3_path": img_3_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": voxel_path,
                        "voxel_1_2_path": voxel_1_2_path,
                        "stereo_depth_path": stereo_depth_path
                    })
                
        return scans  # return to self.data_infos
    
    def get_ann_info(self, index, key='voxel_path'):
        info = self.data_infos[index][key]
        return None if info is None else np.load(info)
    
    @staticmethod
    def read_calib(calib_path):
        """calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
            matrices after rectification. Here P0 denotes the left and P1 denotes the
            right camera. Tr transforms a point from velodyne coordinates into the
            left rectified camera coordinate system. In order to map a point X from the
            velodyne scanner to a point x in the i'th image plane, you thus have to
            transform it like:
            x = Pi * Tr * X
            - 'image_00': left rectified grayscale image sequence
            - 'image_01': right rectified grayscale image sequence
            - 'image_02': left rectified color image sequence
            - 'image_03': right rectified color image sequence
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
        calib_out["P2"] = np.identity(4)  # 4x4 matrix
        calib_out["P3"] = np.identity(4)  # 4x4 matrix
        calib_out["P2"][:3, :4] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"][:3, :4] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4) 
        
        return calib_out
    
    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
    
    def dynamic_baseline(self, infos):
        P3 = infos['P3']
        P2 = infos['P2']
        baseline = P3[0,3]/(-P3[0,0]) - P2[0,3]/(-P2[0,0])
        return baseline

import torch
from PIL import Image
from torchvision import transforms
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_SemanticKitti(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """
    def __init__(self, 
            data_config,
            is_train=False,
            img_norm_cfg=None,
            load_stereo_depth=False,
            color_jitter=(0.4, 0.4, 0.4)
        ):
        super().__init__()

        self.is_train = is_train
        self.data_config = data_config
        self.img_norm_cfg = img_norm_cfg

        self.load_stereo_depth = load_stereo_depth
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )

        self.normalize_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.ToTensor = transforms.ToTensor()

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])

        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        
        return img
    
    def get_inputs(self, results, flip=None, scale=None):
        img_filenames = results['img_filename']

        focal_length = results['focal_length']
        baseline = results['baseline']

        data_lists = []
        raw_img_list = []
        for i in range(len(img_filenames)):
            img_filename = img_filenames[i]
            img = Image.open(img_filename).convert('RGB')

            # perform image-view augmentation
            post_rot = torch.eye(2)
            post_trans = torch.zeros(2)

            if i == 0:
                img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_trans, resize=resize, 
                resize_dims=resize_dims, crop=crop, flip=flip, rotate=rotate
            )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # intrins
            intrin = torch.Tensor(results['cam_intrinsic'][i])

            # extrins
            lidar2cam = torch.Tensor(results['lidar2cam'][i])
            cam2lidar = lidar2cam.inverse()
            rot = cam2lidar[:3, :3]
            tran = cam2lidar[:3, 3]

            # output
            canvas = np.array(img)

            if self.color_jitter and self.is_train:
                img = self.color_jitter(img)
            
            img = self.normalize_img(img)
            depth = torch.zeros(1)

            result = [img, rot, tran, intrin, post_rot, post_tran, depth, cam2lidar]
            result = [x[None] for x in result]

            data_lists.append(result)
            raw_img_list.append(canvas)
        
        if self.load_stereo_depth:
            stereo_depth_path = results['stereo_depth_path']
            stereo_depth = np.load(stereo_depth_path)
            stereo_depth = Image.fromarray(stereo_depth)
            resize, resize_dims, crop, flip, rotate = img_augs
            stereo_depth = self.img_transform_core(stereo_depth, resize_dims=resize_dims,
                    crop=crop, flip=flip, rotate=rotate)
            results['stereo_depth'] = self.ToTensor(stereo_depth)
        num = len(data_lists[0])
        result_list = []
        for i in range(num):
            result_list.append(torch.cat([x[i] for x in data_lists], dim=0))
        
        result_list.append(torch.tensor(focal_length, dtype=torch.float32))
        result_list.append(torch.tensor(baseline, dtype=torch.float32))
        results['raw_img'] = raw_img_list

        return result_list

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)

        return results
    
def data_congif():
    data_root = '/u/home/caoh/datasets/SemanticKITTI/dataset'
    ann_file = '/u/home/caoh/datasets/SemanticKITTI/dataset/labels'
    stereo_depth_root = '/u/home/caoh/datasets/SemanticKITTI/dataset/sequences_msnet3d_depth'
    camera_used = ['left']
    # camera_used = ['left']

    dataset_type = 'SemanticKITTIDataset'
    point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
    occ_size = [256, 256, 32]
    lss_downsample = [2, 2, 2]

    voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
    voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
    voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
    voxel_size = [voxel_x, voxel_y, voxel_z]

    grid_config = {
        'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
        'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
        'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
        'dbound': [2.0, 58.0, 0.5],
    }

    empty_idx = 0

    semantic_kitti_class_frequencies = [
            5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05,
            8.21951000e05, 2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07,
            4.50296100e06, 4.48836500e07, 2.26992300e06, 5.68402180e07, 1.57196520e07,
            1.58442623e08, 2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
        ]

    # 20 classes with unlabeled
    class_names = [
        'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
        'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
        'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
        'pole', 'traffic-sign',
    ]
    num_class = len(class_names)

    # dataset config #
    bda_aug_conf = dict(
        rot_lim=(-22.5, 22.5),
        scale_lim=(0.95, 1.05),
        flip_dx_ratio=0.5,
        flip_dy_ratio=0.5,
        flip_dz_ratio=0
    )

    data_config={
        'input_size': (384, 1280),
        # 'resize': (-0.06, 0.11),
        # 'rot': (-5.4, 5.4),
        # 'flip': True,
        'resize': (0., 0.),
        'rot': (0.0, 0.0 ),
        'flip': False,
        'crop_h': (0.0, 0.0),
        'resize_test': 0.00,
    }
    
    train_pipeline = [
        dict(type='LoadMultiViewImageFromFiles_SemanticKitti', data_config=data_config, load_stereo_depth=True,
            is_train=True, color_jitter=(0.4, 0.4, 0.4)),
        dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti'),
        dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf, apply_bda=False,
                is_train=True, point_cloud_range=point_cloud_range),
        dict(type='CollectData', keys=['img_inputs', 'gt_occ'], 
                meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth', 'gt_occ_1_2']),
    ]

    trainset_config=dict(
        type=dataset_type,
        stereo_depth_root=stereo_depth_root,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split='train',
        camera_used=camera_used,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        test_mode=False,
    )
    
    return trainset_config

if __name__ == '__main__':
    trainset_config = data_congif()
    
    print(trainset_config)
    
    train_dataset = build_dataset(trainset_config)
    
    print('=================')
    
    print(train_dataset)