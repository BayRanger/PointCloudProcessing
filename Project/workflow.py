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
    visualize_kitti(point_cloud)
  
# %%
