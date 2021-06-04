# %%
import numpy as np
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import operator
import matplotlib.pyplot as plt
import copy
from iss import *
from tools import IO
from tools import ISS
from tools import FPFH
from tools import RANSAC
from scipy.spatial.distance import pdist
import random

def parse_bin_to_pcd(filename):
    """Extract pointcloud information from the raw .bin file
    and delete the norm information

    Parameters
    ----------
    filename : [string]
        full path of the bin file

    Returns
    -------
    PointCloud
        the poincloud information
    """
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    data_np = np.fromfile(filename, dtype=np.float32)
    #data_np =  np.asarray(point_cloud_o3d.points)
    #print(np.shape(data_np))
    all_data = (np.reshape(data_np, (int(data_np.shape[0]/6), 6)))

    point_cloud_o3d.points =  o3d.utility.Vector3dVector(all_data[:,0:3])
    point_cloud_o3d.normals =  o3d.utility.Vector3dVector((all_data[:,3:6]))

    return point_cloud_o3d

def draw_registration_result(source, target, transformation=np.diag([1,1,1,1])):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
    
def downsample_pcd(pcd,voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down
    

    
def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])
    
def fpfh_extractor(pcd,voxel_size):
    """extract FPFH features 

    Parameters
    ----------
    pcd : [type]
        [description]
    voxel_size : [type]
        [description]

    Returns
    -------
    [type] np.array
        [description] [33,N]
    """
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh

def get_match_candidate(src_fpfh_data, tgt_fpfh_data,max_dist=20):
    #store the 
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_fpfh_data)
    src_tree = o3d.geometry.KDTreeFlann(src_fpfh_data)
    _,N = np.shape(tgt_fpfh_data)
    _,N_src = np.shape(src_fpfh_data)
    src_match_dict={}
    tgt_match_dict={}
    matches=[]
    for i in range(N):
        _,tgt_nn_idxs,_ = tgt_tree.search_knn_vector_xd(src_fpfh_data[:,i],1)
        src_match_dict[i]=tgt_nn_idxs[0]
    for i in range(N_src):
        _,src_nn_idxs,_ = src_tree.search_knn_vector_xd(tgt_fpfh_data[:,i],1)
        tgt_match_dict[i]=src_nn_idxs[0]
        
    for key in src_match_dict:
        idx = src_match_dict[key]
        if (tgt_match_dict[idx]==key and np.linalg.norm(src_fpfh_data[:,key]-tgt_fpfh_data[:,idx])<max_dist):
            matches.append([key,idx])       
        # for idx in src_match_dict[key]:
        #     if (key in tgt_match_dict[idx]):
        #         matches.append([key,idx])
        
    return matches
    
def solve_icp(P, Q):
    """
    Solve ICP

    Parameters
    ----------
    P: numpy.ndarray
        source point cloud as N-by-3 numpy.ndarray
    Q: numpy.ndarray
        target point cloud as N-by-3 numpy.ndarray

    Returns
    ----------
    T: transform matrix as 4-by-4 numpy.ndarray
        transformation matrix from one-step ICP

    """
    # compute centers:
    up = P.mean(axis = 0)
    uq = Q.mean(axis = 0)

    # move to center:
    P_centered = P - up
    Q_centered = Q - uq

    U, s, V = np.linalg.svd(np.dot(Q_centered.T, P_centered), full_matrices=True, compute_uv=True)
    R = np.dot(U, V)
    t = uq - np.dot(R, up)

    # format as transform:
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1.0

    return T
    # %%
if __name__ == "__main__":
    
#step 1 数据读取
    #load data
    idx1="1"
    idx2="456"
    filename1 = ("/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/{}.bin").format(idx1)
    filename2 = "/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/"+idx2+".bin"
    src_pcd = parse_bin_to_pcd(filename1)
    tgt_pcd = parse_bin_to_pcd(filename2)
    #draw_registration_result(src_pcd,tgt_pcd)
    #point_cloud_o3d = src_pcd.to_instance("open3d", mesh=False)

    voxel_size =1
    src_down = downsample_pcd(src_pcd,1)
    tgt_down = downsample_pcd(tgt_pcd,1)
    #draw_registration_result(src_down,tgt_down)

# %% step 2 denoise and downsample
    

# denoise the data
    # src_denoise_pcd, _ = src_pcd.remove_radius_outlier(nb_points=4, radius=0.5)
    # tgt_denoise_pcd, _ = tgt_pcd.remove_radius_outlier(nb_points=4, radius=0.5)
    # #draw_registration_result(src_denoise_pcd,tgt_denoise_pcd)


# extract feature cloud
    #iss_idxs =getIssKeyPointIdxFromPcd(src_pcd)
    #print(iss_idxs)
    src_iss_pcd = extract_iss_pcd(src_down)
    tgt_iss_pcd = extract_iss_pcd(tgt_down)
    #draw_registration_result(src_iss_pcd,tgt_iss_pcd)
    
# FPFH descriptor
    src_iss_fpfh = fpfh_extractor(src_iss_pcd,voxel_size)
    tgt_iss_fpfh = fpfh_extractor(tgt_iss_pcd,voxel_size)
    print( np.shape(src_iss_fpfh.data))
# %%
#match features 
    src_fpfh_data = src_iss_fpfh.data
    tgt_fpfh_data = tgt_iss_fpfh.data
# # %%
    #do a simple match
    #tgt_iss_fpfh

    match_candidates = get_match_candidate(src_fpfh_data,tgt_fpfh_data)
    match_candidates = np.asarray(match_candidates,dtype=np.int)
    n,_ = np.shape(match_candidates)
    print(np.shape(match_candidates))
    # the matches of points in isis are acquired
    src_iss_data =np.asarray(src_iss_pcd.points)
    tgt_iss_data =np.asarray(tgt_iss_pcd.points)

    print(match_candidates[:,0])
    src_matcher =src_iss_data[match_candidates[:,0]]
    tgt_matcher =tgt_iss_data[match_candidates[:,1]]

# %%
    d_thre =3
    max_matcher =0
    best_T=np.diag([1]*4)
    # select right matches
    for i in range(6000):
        proposal =random.sample(range(0, n), 3)
        A=src_matcher[proposal]
        B=tgt_matcher[proposal]
        T =solve_icp(A,B)    
        R, t = T[0:3, 0:3], T[0:3, 3]
        dist = np.linalg.norm(src_iss_data@R.T+t-tgt_iss_data,axis=1)
        match_num=np.sum(dist<d_thre)
        if(match_num>max_matcher):
            best_T = T
            max_matcher = match_num
    print("best T is ",best_T,"max matchers are ",max_matcher)
        
        
    draw_registration_result(src_down,tgt_down,best_T)


 
        
    

# store the tgt data into the tree

 
# %%
