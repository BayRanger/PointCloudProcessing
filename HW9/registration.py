# -*- coding: UTF-8 -*-


# %%
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import operator
import matplotlib.pyplot as plt

# %%

# def visual_bin(filename):
#     point_cloud_pynt = PyntCloud.from_file(filename)
#     point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
#     pcd = point_cloud_o3d.voxel_down_sample(2)
#     o3d.visualization.draw_geometries([pcd])
#     return pcd

def getOriginPcd(filename):
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    return point_cloud_o3d
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def VisualizePcd(pcd):
    o3d.visualization.draw_geometries([pcd])

def read_oxford_bin(bin_path):
    '''
    :param path:
    :return: [x,y,z,nx,ny,nz]: 6xN
    '''
    data_np = np.fromfile(bin_path, dtype=np.float32)
    return np.transpose(np.reshape(data_np, (int(data_np.shape[0]/6), 6)))
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def get_matched_id(src_idx,source_fpfh,target_fpfh):
    source_fpfh_sample= source_fpfh[:,[src_idx]]
    feature_diff = np.linalg.norm(target_fpfh - source_fpfh_sample,axis=0)
    match_id = np.argmin(feature_diff)
    return match_id,feature_diff[match_id]
    
# %%
if __name__ == "__main__":
    
    #load data
    filename1 = "/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/"+"0.bin"
    filename2 = "/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/"+"1.bin"
    source_pcd = getOriginPcd(filename1)
    target_pcd = getOriginPcd(filename2)
    #source_np =read_oxford_bin(filename1)
    #target_np =read_oxford_bin(filename2)
    

    # downsample and get FPFH
    voxel_size =2
    source_down, source_fpfh = preprocess_point_cloud(source_pcd,voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    source_fpfh_data =source_fpfh.data
    target_fpfh_data = target_fpfh.data
    #output =execute_global_registration(source_down, target_down, source_fpfh, target_fpfh,voxel_size)
    # VisualizePcd(output)
    # get the minmum norm diff
    # source_fpfh_sample= source_fpfh_data[:,[4]]
    # feature_diff = np.linalg.norm(target_fpfh_data - source_fpfh_sample,axis=0)
    # match_id = np.argmin(feature_diff)
    #print(target_fpfh_data[:,match_id])
    #select 10numbers
    # %%
    src_pt_num = np.shape(source_fpfh_data)[1]
    idxs_set = np.random.randint(0,src_pt_num,10)
    dict_match = {}
    for idx in idxs_set:
        print(idx)
        matchid,match_diff = get_matched_id(idx,source_fpfh_data,target_fpfh_data)        
        dict_match[idx]=(matchid,match_diff)
    print(dict_match)
    result = sorted(dict_match.items(),key=lambda x:x[1])
    # %%
    first_three= list(result)[:3]
    src_idx = list(map(lambda x:x[0],first_three))
    target_idx= list(map(lambda x:x[1][0],first_three))
        #dict_match[idx]["diff"] = match_diff
        
    #for i in range(np.shape(source_fpfh_data)[1]):
        

 

# %%
