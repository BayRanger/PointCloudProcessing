# %%
import glob
import os
from os.path import split
import matplotlib.pyplot as plt
import shutil

from pyntcloud.io import pcd
import progressbar
import pandas as pd
import numpy as np
import open3d as o3d
import scipy

def plot_counter(N_pedestrian, N_cyclist, N_vehicle, N_misc):
    plt.bar(["Pedestrian","Cyclist","Vechicle","Misc"],[N_pedestrian, N_cyclist, N_vehicle, N_misc])
    plt.xlabel("Object Type")
    plt.ylabel("Amount")
    plt.show()

def create_folder(output_dir):
    if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    
def random_rot(points):
    N = points.shape[0]

    points_ = points[:,0:3]
    normals_ = points[:,3:6]

    weights = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(points_, 'euclidean')
    ).mean(axis = 0)
    weights /= weights.sum()
    
    idx = np.random.choice(
        np.arange(N), 
        size = (64, ), replace=True if 64 > N else False,
        p = weights 
    )
    points_processed, normals_processed = points_[idx], normals_[idx]
    points = np.concatenate((points_processed,normals_processed),axis=1)
    theta = np.random.uniform(0, np.pi * 2)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rot_points = points
    rot_points[:,0:2] = rot_points[:,0:2].dot(rotation_matrix)
    rot_points[:,3:5] = rot_points[:,3:5].dot(rotation_matrix)
    return rot_points

def read_point_cloud(path):
    
        points = pd.read_csv(path)
        points = np.asarray(points)

        return points
    
def data_aug(path_class,N_aug,output_path):
    print(N_aug)
    progress = progressbar.ProgressBar()
    #print("path_class ",path_class)
    cls = path_class[0].split("/")[1]
    index = 0
    for path in progress(path_class):
        # 读取点云
        points_source = read_point_cloud(path)
        if points_source.shape[0] < 4:
            continue
        for _ in range(N_aug):
            points_random = random_rot(points_source)
            N,_ =np.shape(points_random[:,:])
            mean=(0,0,0)
            cov=np.diag([0.2,0.2,0.1])
            noise = np.random.multivariate_normal(mean, cov, np.shape(N))
            points_random[:,:3]= points_random[:,:3]+noise
            pd_points = pd.DataFrame(points_random)
            pd_points.to_csv(os.path.join(output_path,cls,f'{index:06d}.txt'),index=False, header=None)
            index = index + 1
                    
               
def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])         
if __name__ == "__main__":
    os.chdir(r"/home/chahe/project/PointCloud3D/PointCloudProcessing/Project")

    #output_dir = "/home/teamo/point_process_piplines/KITTI/object/object_training_datasets"
    output_dir = "training_dataset"

    dataset_dir = "training_dataset"
    path_pedestrian = glob.glob(os.path.join(dataset_dir,"pedestrian","*.txt"))
    path_cyclist = glob.glob(os.path.join(dataset_dir,"cyclist","*.txt"))
    path_vehicle = glob.glob(os.path.join(dataset_dir,"vehicle","*.txt"))
    path_misc = glob.glob(os.path.join(dataset_dir,"misc","*.txt"))
    N_pedestrian = len(path_pedestrian)
    N_cyclist = len(path_cyclist)
    N_vehicle = len(path_vehicle)
    N_misc = len(path_misc)
    #print(N_pedestrian)
    # %%
    plot_counter(N_pedestrian, N_cyclist, N_vehicle, N_misc)
    create_folder(output_dir)
    N_aug_pedestrian = int(N_vehicle/N_pedestrian)
    N_aug_cyclist = int(N_vehicle/N_cyclist)
    N_aug_misc = int(N_vehicle/N_misc)
    
    os.mkdir(os.path.join(output_dir,"pedestrian"))
    data_aug(path_pedestrian,N_aug_pedestrian,output_path=output_dir)
    os.mkdir(os.path.join(output_dir,"cyclist"))
    data_aug(path_cyclist,N_aug_cyclist,output_path=output_dir)
    os.mkdir(os.path.join(output_dir,"misc"))
    data_aug(path_misc,N_aug_misc,output_path=output_dir)
    os.mkdir(os.path.join(output_dir,"vehicle"))      
    data_aug(path_vehicle,1,output_path=output_dir)
    #visualize_pcd(pcd_original)
    
# %%
