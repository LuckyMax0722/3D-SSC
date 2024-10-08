import open3d as o3d
import numpy as np

# 读取 bin 文件
def read_bin(bin_path):
    point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # 每个点 (x, y, z, intensity)
    return point_cloud

# 可视化点云
def visualize_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # 只保留 x, y, z
    # 可以根据需要添加颜色等属性
    o3d.visualization.draw_geometries([pcd])

# 加载并显示点云
bin_file = '/media/max/GAME/MA/datasets/SemanticKITTI/dataset/sequences_msnet3d_lidar/sequences/00/demo.bin'  # 替换为你生成的 .bin 文件路径

#bin_file = '/media/max/GAME/MA/datasets/SemanticKITTI/dataset/sequences_msnet3d_lidar_msn/sequences/00/000000.bin'  # 替换为你生成的 .bin 文件路径


point_cloud_data = read_bin(bin_file)
visualize_point_cloud(point_cloud_data)
