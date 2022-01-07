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
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from multiprocessing import cpu_count
import time


def test():
    for i in progressbar.progressbar(range(100)):
        time.sleep(0.02)
   
def transform_from_velo_to_obj(X_velo, param, t_obj_to_cam, ry):
    
    # get params:
    R0_rect = param['R0_rect']
    R_velo_to_cam, t_velo_to_cam = param['Tr_velo_to_cam'][:,0:3], param['Tr_velo_to_cam'][:,3]

    # project to unrectified camera frame:
    X_cam = np.dot(
        R_velo_to_cam, X_velo.T
    ).T + t_velo_to_cam

    # rectify:
    X_cam = np.dot(
       R0_rect, X_cam.T
    ).T

    # project to object frame:
    cos_ry = np.cos(ry)
    sin_ry = np.sin(ry)

    R_obj_to_cam = np.asarray(
        [
            [ cos_ry, 0.0, sin_ry],
            [    0.0, 1.0,    0.0],
            [-sin_ry, 0.0, cos_ry]
        ]
    )

    X_obj = np.dot(
        R_obj_to_cam.T, (X_cam - t_obj_to_cam).T
    ).T

    return X_obj
     
def read_lidar_calib_label(data_dir,index):
    point_cloud = (os.path.join(data_dir, 'velodyne', f'{index:06d}.bin'))
    param = read_calib(os.path.join(data_dir, 'calib', f'{index:06d}.txt'))
    df_label = read_label(os.path.join(data_dir, 'label_2', f'{index:06d}.txt'),param) 
    return point_cloud,df_label

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


def read_calib(filepath):
    DIMENSION = {
        'P0': (3, 4),
        'P1': (3, 4),
        'P2': (3, 4),
        'P3': (3, 4),
        'R0_rect': (3, 3),
        'Tr_velo_to_cam': (3, 4),
        'Tr_imu_to_velo': (3, 4)
    }

    param = {}
    # parse calibration data:
    with open(filepath, 'rt') as f:
        # one line per param:
        content = [tuple(i.split(':')) for i in f.read().strip().split('\n')]
        # format param as numpy.ndarray with correct shape
        for name, value in content:
            param[name] = np.asarray(
                [float(v) for v in value.strip().split()]
            ).reshape(
                DIMENSION[name]
            )
    
    return param


def transform_from_cam_to_velo(X_cam, param):
    # get params:
    R0_rect = param['R0_rect']
    R_velo_to_cam, t_velo_to_cam = param['Tr_velo_to_cam'][:,0:3], param['Tr_velo_to_cam'][:,3]

    # unrectify:
    X_velo = np.dot(
        R0_rect.T, X_cam.T
    ).T

    # project to velo frame:
    X_velo =(
        R_velo_to_cam.T@(X_velo - t_velo_to_cam).T
    ).T

    return X_velo

def read_label(filepath, param):
    
    # load data:    
    df_label = pd.read_csv(
        filepath,
        sep = ' ', header=None
    )

    # add attribute names:
    df_label.columns = [
        'type',
        'truncated',
        'occluded',
        'alpha',
        'left', 'top', 'right', 'bottom',
        'height', 'width', 'length',
        'cx', 'cy', 'cz', 'ry'
    ]

    # filter label with no dimensions:
    condition = (
        (df_label['height'] >= 0.0) &
        (df_label['width'] >= 0.0) &
        (df_label['length'] >= 0.0)
    )
    df_label = df_label.loc[
        condition, df_label.columns
    ]

    #
    # get object center in velo frame:
    #
    centers_cam = df_label[['cx', 'cy', 'cz']].values
    centers_velo = transform_from_cam_to_velo(centers_cam, param)
    # add height bias:
    df_label['vx'] = df_label['vy'] = df_label['vz'] = 0.0
    df_label[['vx', 'vy', 'vz']] = centers_velo
    df_label['vz'] += df_label['height']/2

    # add radius for point cloud extraction:
    df_label['radius'] = df_label.apply(
        lambda x: np.linalg.norm(
            0.5*np.asarray(
                [x['height'], x['width'], x['length']]
            )
        ),
        axis = 1
    )

    return df_label
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
    
def get_object_pcd_df(pcd, idx, num_sample):            
    df_point_cloud_with_normal = pd.DataFrame(
        data = np.hstack(
            (
                np.asarray(pcd.points)[idx],
                np.asarray(pcd.normals)[idx]
            )
        ),
        index = np.arange(N),
        columns = ['vx', 'vy', 'vz', 'nx', 'ny', 'nz']
    )

    return df_point_cloud_with_normal
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

# acquire the non-maximum suppression
def filter_by_bounding_box(X, labels, dims):
    

    # filter by bounding box in object frame:
    idx_obj = np.all(
        np.logical_and(
            X >= -dims/2,
            X <=  dims/2
        ),
        axis = 1
    )

    if idx_obj.sum() == 0:
        return None

    # get object ID using non-maximum suppression:
    ids, counts = np.unique(
        labels[idx_obj], return_counts=True
    )
    object_id, _ = max(zip(ids, counts), key=lambda x:x[1]) 

    return object_id


def get_object_category(object_type):
    
    category = 'vehicle'

    if object_type is None or object_type == 'Misc' or object_type == 'DontCare':
        category = 'misc'
    elif object_type == 'Pedestrian' or object_type == 'Person_sitting':
        category = 'pedestrian'
    elif object_type == 'Cyclist':
        category = 'cyclist'


    return category


def add_label(dataset_label, category, label, N, center):
    
    if label is None:
        # kitti category:
        dataset_label[category]['type'].append('Unknown')
        dataset_label[category]['truncated'].append(-1)
        dataset_label[category]['occluded'].append(-1)

        # bounding box labels:
        dataset_label[category]['height'].append(-1)
        dataset_label[category]['width'].append(-1)
        dataset_label[category]['length'].append(-1)
        dataset_label[category]['ry'].append(-10)
    else:
        # kitti category:
        dataset_label[category]['type'].append(label['type'])
        dataset_label[category]['truncated'].append(label['truncated'])
        dataset_label[category]['occluded'].append(label['occluded'])

        # bounding box labels:
        dataset_label[category]['height'].append(label['height'])
        dataset_label[category]['width'].append(label['width'])
        dataset_label[category]['length'].append(label['length'])
        dataset_label[category]['ry'].append(label['ry'])

    # distance and num. of measurements:
    dataset_label[category]['num_measurements'].append(N)
    vx, vy, vz = center
    dataset_label[category]['vx'].append(vx)
    dataset_label[category]['vy'].append(vy)
    dataset_label[category]['vz'].append(vz)

def process(index):
    param = read_calib(os.path.join(data_dir, 'calib', f'{index:06d}.txt'))

        # 读取label文件
    df_label = read_label(os.path.join(data_dir, 'label_2', f'{index:06d}.txt'),param)
        
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
    search_tree = o3d.geometry.KDTreeFlann(segmented_objects)

    #visualize_pcd(segmented_objects)

    search_tree = o3d.geometry.KDTreeFlann(segmented_objects)
    identified = set()
    for idx, label in df_label.iterrows():
        #print("idx ",idx," label ",label)
        # 提取对应的参数
        center_velo = np.asarray([label['vx'], label['vy'], label['vz']])

        #print(np.linalg.norm(center_velo))

        if np.linalg.norm(center_velo) > max_distance:
            continue
        
        center_cam = np.asarray([label['cx'], label['cy'], label['cz']])

        # dimensions in camera frame:
        dims = np.asarray([label['length'], label['height'], label['width']])
        

        [k, idx, _] = search_tree.search_radius_vector_3d(
            center_velo, 
            label['radius']
        )

        if (k > 0):     
            point_cloud_velo_ = np.asarray(segmented_objects.points)[idx]
            object_ids_ = object_ids[idx]


            point_cloud_obj = transform_from_velo_to_obj(
                point_cloud_velo_, 
                param, 
                center_cam, 
                label['ry']
            )

            # add bias along height:
            point_cloud_obj[:, 1] += label['height']/2


            # 进行矩形框滤波，并获取对应的聚类id（在矩形框内点数最多的类别的id）
            object_id_ = filter_by_bounding_box(point_cloud_obj, object_ids_, dims)

            if object_id_ is None:
                continue
                
            identified.add(object_id_)

            # 获取对应聚类的点云id
            idx_object = np.asarray(idx)[object_ids_ == object_id_]


            # 构建对应的点云dataframe
            N = len(idx_object)
            df_point_cloud_with_normal = get_object_pcd_df(segmented_objects, idx_object, N)

            # 获取类别:
            category = get_object_category(label['type'])
            # 获取对应点云的中心
            center = np.asarray(segmented_objects.points)[idx_object].mean(axis = 0)
            #print("center",label)

            add_label(dataset_label, category, label, N, center)
            #print("dataset label",dataset_label)
            dataset_index = len(dataset_label[category]['type'])
            df_point_cloud_with_normal.to_csv(
                    os.path.join(output_dir, category, f'{dataset_index:06d}.txt'),
                    index=False, header=None
                )
    for category in ['vehicle', 'pedestrian', 'cyclist', 'misc']:
        dataset_label[category] = pd.DataFrame.from_dict(
            dataset_label[category]
        )
        dataset_label[category].to_csv(
            os.path.join(output_dir, f'{category}.txt'),
            index=False
    )            
if __name__ =="__main__":
    data_dir ="/home/chahe/project/PointCloud3D/PointCloudProcessing/HW6/PointRCNN/data/KITTI/object/training" 
    N = len(glob.glob(os.path.join(data_dir,"label_2","*.txt")))
    print("There are ",N," labeling files")
    output_dir = "output_new"
    os.chdir(r"/home/chahe/project/PointCloud3D/PointCloudProcessing/Project")
    cwd = os.getcwd()
    print("current directory is ",cwd)
    max_distance = 25

    # set output  directoryy
    dataset_label = {}
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
 
    progres = progressbar.ProgressBar()
    for cat in ['vehicle', 'pedestrian', 'cyclist', 'misc']:
        os.mkdir(os.path.join(output_dir,cat))
        dataset_label[cat] = init_label()
    #index=2
    #for index in progres(range(N)):
    
    
    start = time.time()
    # for i in range(10):
    #     write_index(i)
    with ThreadPoolExecutor(16) as ex:
        ex.map(process, np.arange(N))
    end=time.time()
    print("Time cost: ",end-start)
    
    # %%
"""
TODO: use multiprogramming to increase the speed of processing
"""