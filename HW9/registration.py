# -*- coding: UTF-8 -*-


# %%
 
from util import * 
# %% 
    
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


def get_matched_id(src_idx,source_fpfh,target_fpfh):
    source_fpfh_sample= source_fpfh[:,[src_idx]]
    feature_diff = np.linalg.norm(target_fpfh - source_fpfh_sample,axis=0)
    match_id = np.argmin(feature_diff)
    return match_id,feature_diff[match_id]

def procrustes_transformation(A, B):
    """ Procrustes was a rogue smith and bandit from Attica who attacked people 
    by stretching them or cutting off their legs, so as to force them 
    to fit the size of an iron bed.

    Parameters
    ----------
    A : [numpy.array]
        [size:N,3]
    B : [numpy.array]
        [size: N,3]

    Returns
    -------
    R,t
        [description]
        Rotation matrix and transforamtion vector
    """
    N = A.shape[0]
    
    A_mean= np.mean(A,axis=0)
    B_mean = np.mean(B,axis=0)
    Ap = A - A_mean
    Bp = B - B_mean
    u, s, vt = np.linalg.svd(Bp.T@Ap)
    #sR= vt.T@u.T
    R= u@vt

    t= np.expand_dims(B_mean - R@A_mean,1)
    
    return R,t
        
# %%
if __name__ == "__main__":
    
    #load data
    filename1 = "/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/"+"0.bin"
    filename2 = "/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/"+"456.bin"
    source_pcd = getCuttedPcd(filename1)
    target_pcd = getCuttedPcd(filename2)
    #test_pcd =io.read_point_cloud_bin(filename1)
    #source_pcd.points
    # %%
     #VisualizePcd(source_pcd)
    #VisualizePcd(target_pcd)
    src_data = read_oxford_bin(filename1)[:3]
    tgt_data = read_oxford_bin(filename2)[:3]
    # src_features, tgt_features: (length of feature, #points)
    voxel_size =2
    source_down, source_fpfh = preprocess_point_cloud(source_pcd,voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    source_data = np.asarray(source_down.points)
    target_data = np.asarray(target_down.points)
    src_features = np.asarray(source_fpfh.data)
    tgt_features = np.asarray(target_fpfh.data)
    src_tree = o3d.geometry.KDTreeFlann(src_features)
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_features)
    N_src = src_features.shape[1]
    N_tgt = tgt_features.shape[1]
    dist_thres = 1 # a guess
    #print(dir(src_tree))
    # %%
    # get matchers
    #print(dir(src_tree))
    #build two dicts to store features
    src_match_dict ={}
    tgt_match_dict ={}
    matchings = []
    #load the dictionary
    for idx in range(N_src):
        src_feature = src_features[:,idx]
        _,match_idxs,_ =tgt_tree.search_knn_vector_xd(src_feature,1)
        src_match_dict[idx]=np.array(match_idxs)
    for idx in range(N_tgt):
        tgt_feature = tgt_features[:,idx]
        _,match_idxs,_ =src_tree.search_knn_vector_xd(tgt_feature,1)
        tgt_match_dict[idx]=np.array(match_idxs)
    #match the dictionary
    for key in src_match_dict:
        for idx in src_match_dict[key]:
            if (key in tgt_match_dict[idx]):
                #print("find a pair",key,", ",idx)
                matchings.append((key,idx))
                break
    # get the best  R, T after iteration
# %%
    max_iter= 1
    sample_num=9
    pairs_num =(len(matchings))
# %%
    A_in_order =np.zeros((pairs_num,3))
    B_in_order =np.zeros((pairs_num,3))
    for i in range(pairs_num):
        src_idx = matchings[i][0]
        tgt_idx = matchings[i][1]
        A_in_order[i]=source_data[src_idx]
        B_in_order[i]=target_data[tgt_idx]       
    
# %%
    best_R=None
    best_t=None
    min_res= 100000
    max_iter=4000
    for i in range(max_iter):
        A_tmp = np.zeros((sample_num,3))
        B_tmp = np.zeros((sample_num,3))
        matches_set = np.random.randint(0,pairs_num,sample_num)
        # build random sample
        for j in range(sample_num):
            src_idx = matchings[matches_set[j]][0]
            tgt_idx = matchings[matches_set[j]][1]
            A_tmp[j]=source_data[src_idx]
            B_tmp[j]=target_data[tgt_idx]
        # init registration
        # 1R,t,residual =procrustes_transformation(A_tmp,B_tmp)
        #print("R",R,"t",t)
        A = A_tmp
        B = B_tmp
        N = A.shape[0]
        
        A_mean= np.mean(A,axis=0)
        B_mean = np.mean(B,axis=0)
        Ap = A - A_mean
        Bp = B - B_mean
        u, s, vt = np.linalg.svd(Bp.T@Ap)
        #sR= vt.T@u.T
        R= u@vt

        t= np.expand_dims(B_mean - R@A_mean,1)
        new_A =(R@(A_in_order.T)+t).T
        diff = np.linalg.norm(new_A - B_in_order)

        if (diff<min_res):
            best_R=R
            best_t=t
            min_res = diff
    print("best R ",best_R)
    print("best t ",best_t)
    # Observation, the t is always not correct after coarse optimization
            
 

 

# %%
