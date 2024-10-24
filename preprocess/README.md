# SemanticKITTI

!!!Warning!!!

Please set the path in `./SGN/projects/configs/config.py` before you do anything!

## 1. Prepare data
Download the following data from [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) 

* Download odometry data set (color, 65 GB)
* Download odometry data set (calibration files, 1 MB)
* Download odometry ground truth poses (4 MB)

and also from [SemanticKITTI](https://www.semantic-kitti.org/dataset.html)
* Download to get the SemanticKITTI voxel data (700 MB)

also label
https://drive.google.com/file/d/1r6RWjPClt9-EBbuOczLB295c00o7pOOP/view?usp=share_link

Unzip all the .zip files and run
```shell
python ./SGN/preprocess/poses/pose_preprocess.py
```

The data is organized in the following format:

```
./SemanticKITTI/dataset/
                └── sequences/
                        ├── 00/
                        │   ├── poses.txt
                        │   ├── calib.txt
                        │   ├── image_2/
                        │   ├── image_3/
                        |   ├── voxels/
                        |         ├ 000000.bin
                        |         ├ 000000.label
                        |         ├ 000000.occluded
                        |         ├ 000000.invalid
                        |         ├ 000005.bin
                        |         ├ 000005.label
                        |         ├ 000005.occluded
                        |         ├ 000005.invalid
                        ├── 01/
                        ├── 02/
                        .
                        └── 21/

```

## 2. Generating grounding truth
Setting up the environment
```shell
conda create -n preprocess python=3.7 -y
conda activate preprocess
conda install numpy tqdm pyyaml imageio
```
Preprocess the data to generate labels at a lower scale:
```shell
python ./SGN/preprocess/label/label_preprocess.py
```


Then we have the following data:
```
./kitti/dataset/
          └── sequences/
          │       ├── 00/
          │       │   ├── poses.txt
          │       │   ├── calib.txt
          │       │   ├── image_2/
          │       │   ├── image_3/
          │       |   ├── voxels/
          │       ├── 01/
          │       ├── 02/
          │       .
          │       └── 21/
          └── labels/
                  ├── 00/
                  │   ├── 000000_1_1.npy
                  │   ├── 000000_1_2.npy
                  │   ├── 000005_1_1.npy
                  │   ├── 000005_1_2.npy
                  ├── 01/
                  .
                  └── 10/

```

## 3. Image to Point Cloud
We use [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2) to obtain the point cloud.

### Prepraration
Please refer to the [Prepraration](https://github.com/DepthAnything/Depth-Anything-V2#prepraration)

### Prediction

The following script could create point cloud for all sequences:
```shell
python ./SGN/preprocess/depthanythingv2/depth_to_pointcloud_kitti.py
```

Then we have the following data:
```
./kitti/dataset/
          └── sequences/
          │       ├── 00/
          │       │   ├── poses.txt
          │       │   ├── calib.txt
          │       │   ├── image_2/
          │       │   ├── image_3/
          │       |   ├── voxels/
          │       ├── 01/
          │       ├── 02/
          │       .
          │       └── 21/
          └── labels/
          │       ├── 00/
          │       │   ├── 000000_1_1.npy
          │       │   ├── 000000_1_2.npy
          │       │   ├── 000005_1_1.npy
          │       │   ├── 000005_1_2.npy
          │       ├── 01/
          │       .
          │       └── 10/
          └── sequences_msnet3d_lidar/
                  └── sequences
                        ├── 00
                        │   ├ 000001.bin
                        │   ├ 000002.bin
                        ├── 01/
                        ├── 02/
                        .
                        └── 21/
```