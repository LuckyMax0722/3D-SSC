import numpy as np
import pandas as pd
import os
import open3d as o3d


import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from projects.configs.config import CONF

color_map = { # bgr
  0 : [0, 0, 0],
  1 : [0, 0, 255],
  10: [245, 150, 100],
  11: [245, 230, 100],
  13: [250, 80, 100],
  15: [150, 60, 30],
  16: [255, 0, 0],
  18: [180, 30, 80],
  20: [255, 0, 0],
  30: [30, 30, 255],
  31: [200, 40, 255],
  32: [90, 30, 150],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [75, 0, 175],
  50: [0, 200, 255],
  51: [50, 120, 255],
  52: [0, 150, 255],
  60: [170, 255, 150],
  70: [0, 175, 0],
  71: [0, 60, 135],
  72: [80, 240, 150],
  80: [150, 240, 255],
  81: [0, 0, 255],
  99: [255, 255, 50],
  252: [245, 150, 100],
  256: [255, 0, 0],
  253: [200, 40, 255],
  254: [30, 30, 255],
  255: [90, 30, 150],
  257: [250, 80, 100],
  258: [180, 30, 80],
  259: [255, 0, 0]
}

learning_map = {
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car" to "car" ------------------------------------mapped
  253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
  254: 6,    # "moving-person" to "person" ------------------------------mapped
  255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
  256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
  257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
  258: 4,    # "moving-truck" to "truck" --------------------------------mapped
  259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

learning_map_inv = { # inverse of previous map
  0: 0,      # "unlabeled", and others ignored
  1: 10,     # "car"
  2: 11,     # "bicycle"
  3: 15,     # "motorcycle"
  4: 18,     # "truck"
  5: 20,     # "other-vehicle"
  6: 30,     # "person"
  7: 31,     # "bicyclist"
  8: 32,     # "motorcyclist"
  9: 40,     # "road"
  10: 44,    # "parking"
  11: 48,    # "sidewalk"
  12: 49,    # "other-ground"
  13: 50,    # "building"
  14: 51,    # "fence"
  15: 70,    # "vegetation"
  16: 71,    # "trunk"
  17: 72,    # "terrain"
  18: 80,    # "pole"
  19: 81,    # "traffic-sign"
  255: 0
}

def read_npy_file():
    # read npy file 1
    depth_stero = np.load('/media/max/GAME/MA/datasets/SemanticKITTI/dataset/sequences_msnet3d_depth/sequences/00/000000.npy')  # TODO

    df = pd.DataFrame(depth_stero)

    # save to excel
    output_stero = os.path.join(CONF.PATH.PREPROCESS, 'tools', 'output_stero.xlsx')
    df.to_excel(output_stero, index=False)

    # read npy file 2
    depth_mono = np.load('/media/max/GAME/MA/Depth-Anything-V2/metric_depth/output/000000.npy')  # TODO

    df = pd.DataFrame(depth_mono)

    # save to excel
    output_mono = os.path.join(CONF.PATH.PREPROCESS, 'tools', 'output_mono.xlsx')
    df.to_excel(output_mono, index=False)


def read_bin_file():
    bin_path = '/media/max/GAME/MA/datasets/SemanticKITTI/dataset/sequences_msnet3d_lidar/sequences/00/000000.bin'  # TODO
    
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  #  (x, y, z, intensity)
    
    # visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # keep x, y, z
    o3d.visualization.draw_geometries([pcd])

def read_label_npy_file():
    file_path = '/media/max/GAME/MA/SGN/preprocess/tools/000000_1_1_sgn.npy'
    data = np.load(file_path)
    
    # 提取非零体素的坐标
    x, y, z = np.nonzero(data)

    voxel_labels = data[x, y, z]

    # 组合为 (x, y, z) 点
    points = np.vstack((x, y, z)).T

    colors = np.array([color_map[learning_map_inv[int(label)]] for label in voxel_labels]) / 255.0  # 将颜色值归一化为 [0, 1]

    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 设置体素的颜色 (可选)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # 创建 XYZ 轴，原点在 (0, 0, 0)，长度为 50
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50, origin=[0, 0, 0])

    # 可视化
    o3d.visualization.draw_geometries([point_cloud, axis])

def main(vis_depth_npy, vis_pcd_bin, vis_pcd_npy):
    if vis_depth_npy:
        # this tool transform the depth npy file to excel, so that we can check depth information
        # stero is for origional mobilesteronet output, mono is for depth anything 
        read_npy_file()
    
    if vis_pcd_bin:
        # this tool visualize the pcd bin file
        read_bin_file()
        
    if vis_pcd_npy:
        read_label_npy_file()


if __name__ == '__main__':
    vis_depth_npy = False
    vis_pcd_bin = False
    vis_pcd_npy = True
    
    main(vis_depth_npy, vis_pcd_bin, vis_pcd_npy)