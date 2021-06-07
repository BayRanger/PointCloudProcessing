# %%

#remove the ground from the lidar points

#clustering over the remaining points

#classification over the clusters

#report the detection precision-recall for three categories:
import argparse
import glob
import os
from numpy.core.fromnumeric import shape
from numpy.lib.type_check import imag
import progressbar
import shutil
import numpy as np
import struct
import pandas as pd
import open3d as o3d
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import time
import progressbar


def test():
    for i in progressbar.progressbar(range(100)):
        time.sleep(0.02)

def read_velodyne_bin(filepath):
    point_cloud = []
    with open(filepath, 'rb') as f:
        # unpack velodyne frame:
        content = f.read()
        measurements = struct.iter_unpack('ffff', content)
        # parse:
        for i, point in enumerate(measurements):
            x, y, z, intensity = point
            point_cloud.append([x, y, z, intensity])
    # format for output
    point_cloud = np.asarray(point_cloud, dtype=np.float32)

    return point_cloud

def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])

def visualize_kitti(array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array[:,:3])
    o3d.visualization.draw_geometries([pcd])

'''
@description: 为每种标签建立对应的字典
@param {*}
@return {*}
'''
def init_label():
    return {
        # original category:
        'type': [],
        'truncated': [],
        'occluded': [],
        # distance and num. of measurements:
        'vx': [], 
        'vy': [], 
        'vz': [], 
        'num_measurements': [],
        # bounding box labels:
        'height': [], 
        'width': [], 
        'length':[], 
        'ry':[]
    }
    
def segment_ground_and_objects(point_cloud):
    N, _ = point_cloud.shape


    # 根据法向量进行滤波
    # 估计每个点的法向量
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(point_cloud)
    pcd_original.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0, max_nn=9
        )
    )

    # 提取z轴角度大于30度的点
    normals = np.asarray(pcd_original.normals)
    angular_distance_to_z = np.abs(normals[:, 2])
    idx_downsampled = angular_distance_to_z > np.cos(np.pi/4)
    # the road points should be closed to orthogonal to the z axis

    # 使用open3d的函数进行平面分割
    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(point_cloud[idx_downsampled])
    #visualize_pcd(pcd_downsampled)

    ground_model, idx_ground = pcd_downsampled.segment_plane(
        distance_threshold=0.30,
        ransac_n=3,
        num_iterations=1000
    )

    # 提取到的属于地面的点
    segmented_ground = pcd_downsampled.select_by_index(idx_ground)

    # 计算原始点云中每个点与地面的距离
    distance_to_ground = np.abs(
        point_cloud@np.asarray(ground_model[:3]) + ground_model[3]
    )

    # 提取非地面的点
    idx_cloud = distance_to_ground > 0.30

    # limit FOV to front:
    segmented_objects = o3d.geometry.PointCloud()

    # 选取可视范围内的点
    idx_segmented_objects = np.logical_and.reduce(
        [
            idx_cloud,
            point_cloud[:, 0] >=   1.95, point_cloud[:, 0] <=  80.00,
            point_cloud[:, 1] >= -30.00, point_cloud[:, 1] <= +30.00
        ]
    )

    # 最后得到的非地面的点
    segmented_objects.points = o3d.utility.Vector3dVector(
        point_cloud[idx_segmented_objects]
    )
    segmented_objects.normals = o3d.utility.Vector3dVector(
        np.asarray(pcd_original.normals)[idx_segmented_objects]
    )

    # 对地面与非地面的点绘制不同的颜色
    segmented_ground.paint_uniform_color([0.0, 0.0, 0.0])
    segmented_objects.paint_uniform_color([0.5, 0.5, 0.5])

    
    # DBSACN聚类，这里结果的格式为N by 1的数据，-1代表噪声数据
    labels = np.asarray(segmented_objects.cluster_dbscan(eps=0.60, min_points=3))
    print(np.shape(labels))

    return segmented_ground, segmented_objects, labels


if __name__ =="__main__":
    data_dir ="/home/chahe/project/PointCloud3D/PointCloudProcessing/HW6/PointRCNN/data/KITTI/object/training" 
    N = len(glob.glob(os.path.join(data_dir,"label_2","*.txt")))
    print("There are ",N," labeling files")
    output_dir = "output"
    os.chdir(r"/home/chahe/project/PointCloud3D/PointCloudProcessing/Project")
    cwd = os.getcwd()
    print("current directory is ",cwd)

    # set output  directoryy
    dataset_label = {}
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    else:
        os.mkdir(output_dir)
    progres = progressbar.ProgressBar()
    for cat in ['vehicle', 'pedestrian', 'cyclist', 'misc']:
        os.mkdir(os.path.join(output_dir,cat))
        dataset_label[cat] = init_label()
    index=0
        # 读取对应的点云
    # point_cloud = read_velodyne_bin(
    # os.path.join(input_dir, 'velodyne', f'{index:06d}.bin'))

    # # 读取对应的相机矫正文件以及转换矩阵
    # param = read_calib(
    # os.path.join(input_dir, 'calib', f'{index:06d}.txt'))

    # # 读取label文件
    # df_label = read_label(
    # os.path.join(input_dir, 'label_2', f'{index:06d}.txt'),param)
    point_cloud = read_velodyne_bin(os.path.join(data_dir, 'velodyne', f'{index:06d}.bin'))
    #print(f'{index:06d}.txt')
    #visualize_kitti(point_cloud)
    
    segmented_ground, segmented_objects, object_ids = segment_ground_and_objects(point_cloud[:, 0:3])

    #visualize_pcd(segmented_ground)
    visualize_pcd(segmented_objects)

    # search_tree = o3d.geometry.KDTreeFlann(segmented_objects)

# %%
