# -*- coding: UTF-8 -*-


# %%
from scipy.spatial.transform import Rotation  
from util import * 
import numpy
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


def procrustes_transform(A,B,k=3):
    L = np.eye(k) - np.ones([k,k])*1/k
    print("Ap shape",np.shape(L),"A",np.shape(A))
    Ap =np.matmul(A.T,L)
    Bp = np.matmul(B.T,L)
    U,S,V= np.linalg.svd(np.matmul(Bp,Ap.T))
    R=U@V
    t=1/k*(B.T -R@A.T)@np.ones([k,1])
    
    return R,t

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

def getMatchesFromFeature(src_features,tgt_features,src_data,tgt_data):
    """The matches are formed based on the feature descriptors

    Parameters
    ----------
    src_features : [numpy.array m*N]
        [description]
        source feature
    tgt_features : [numpy.array  m*N]
        [description]
        target features
    src_data
    tgt_data    
    

    Returns
    -------
    [list]
        [description]
        A list of matched indexs tuple
        
    """
    src_tree = o3d.geometry.KDTreeFlann(src_features)
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_features)
    N_src = src_features.shape[1]
    N_tgt = tgt_features.shape[1]
 
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
                #TODO: if there are any conditions should be satisfied.
                #print("find a pair",key,", ",idx)
                #if(np.linalg.norm(src_data[key]-tgt_data[idx],2)<4):
                matchings.append((key,idx))
                #break
    return matchings

# %%
def Rt2Homology(R,t):
    H = np.diag([1.0,1.0,1.0,1.0])
    H[:3,:3] = R
    t = t.reshape((3))
    H[:3,3] = t
    return H
    
    
    return R_new,t_new, res

def Rt2string(idx1,idx2,R,t):
    x, y, z, w =Rotation.from_matrix(R).as_quat()
    res="{},{},{},{},{},{},{},{},{}".format(idx1,idx2,t[0][0],t[1][0],t[2][0],w,x,y,z)
    return res
    
def stirng2File(idx1,idx2,R,t):
    out_file = open("/home/chahe/project/PointCloud3D/dataset/registration_dataset/reg_result.txt", "w")
    out_file.write(Rt2string(idx1,idx2,R,t))
    out_file.close()
        
def find_accociations(src_all_data,tgt_all_data,R_prev,t_prev,threshold,tgttree):
    """[summary]

    Parameters
    ----------
    src_pts : numpy.array
        The original data
    R_init : 3*3 numpy.array
        Previous Rotation matrix
    t_init : 3*1 numpy.array
        Previous translation vector
    threshold : float
        Decide if the pair is too far away
    tgttree : Kdtree
        A tree stores all the data information

    Returns
    -------
    [type]
        [description]
    """
    N= np.shape(src_all_data)[1]
    # move the data based on the initial information
    transformed_src_data = R_prev@src_all_data+ t_prev.reshape((3,1))
    # # Try to match every point in the  stc
    # #match_pairs=[]
    # matched_src_data=[]
    # matched_tgt_data=[]
    # for i in range(N):
    #     to_match_pt = transformed_src_data[:,[i]]
    #     _,match_idx,dists =tgt_data_tree.search_knn_vector_xd(to_match_pt,1)
    #     #if (np.linalg,norm(to_match_pt - tgt)
    #     if dists[0]<threshold:
    #         #match_pairs.append([i,match_idx[0]])
    #         matched_src_data.append(transformed_src_data[:,i])
    #         matched_tgt_data.append(tgt_all_data[:,match_idx[0]])
    matched_src_data = transformed_src_data
    matched_tgt_data = tgt_all_data
    print("HHHHHHHHHH",np.shape(matched_src_data))
    num = (np.shape(matched_src_data))[0]
# %%
    new_R,new_t = procrustes_transformation(matched_src_data.T,matched_tgt_data.T)
    R_update = new_R@R_prev
    t_update = np.array(new_t).reshape((3,1))+ np.array(t_prev).reshape((3,1))
    final_q =Rotation.from_matrix(R_update).as_quat()
    # prev_cost= 1.0/N*np.linalg.norm(R_prev@matched_src_data.T+t_prev.reshape((3,1)) - matched_tgt_data.T)
    cost =np.linalg.norm(new_R@matched_src_data+new_t - matched_tgt_data)
    matches =np.linalg.norm(new_R@matched_src_data+new_t - matched_tgt_data,axis=0)
    threshold =1
    match_num =np.sum(matches<threshold)
    print("matches ",match_num)
    #print("final q ",final_q,"  final t",new_t)
    
    
    return R_update,t_update, cost 

    

        
# %%
if __name__ == "__main__":
    
    #load data
    idx1="643"
    idx2="456"
    filename1 = ("/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/{}.bin").format(idx1)
    filename2 = "/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/"+idx2+".bin"
    source_pcd = parse_bin_to_pcd(filename1)
    target_pcd = parse_bin_to_pcd(filename2)
    src_all_data = np.asarray(source_pcd.points).T
    tgt_all_data = np.asarray(target_pcd.points).T
    #test_pcd =io.read_point_cloud_bin(filename1)

    # %%
    voxel_size =1
    source_down, source_fpfh = preprocess_point_cloud(source_pcd,voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    source_data = np.asarray(source_down.points)
    target_data = np.asarray(target_down.points)
    src_features = np.asarray(source_fpfh.data)
    tgt_features = np.asarray(target_fpfh.data)
    matchings =getMatchesFromFeature(src_features, tgt_features,source_data,target_data)


# %%
    sample_num=5
    pairs_num =len(matchings)
    A_in_order =np.zeros((pairs_num,3))
    B_in_order =np.zeros((pairs_num,3))
    for i in range(pairs_num):
        src_idx = matchings[i][0]
        tgt_idx = matchings[i][1]
        A_in_order[i]=source_data[src_idx]
        B_in_order[i]=target_data[tgt_idx]       
    
# %%
    init_R=None
    init_t=None
    match_max=0
    max_iter=4000
    d_thre=0.5
    for i in range(max_iter):
        A_tmp = np.zeros((sample_num,3))
        B_tmp = np.zeros((sample_num,3))
        matches_set = np.random.randint(0,pairs_num,sample_num)
        # build random sample
        for j in range(sample_num):
            A_tmp[j]=A_in_order[matches_set[j]]
            B_tmp[j]=B_in_order[matches_set[j]]

        R,t = procrustes_transformation(A_tmp,B_tmp)
        new_A =(R@(A_in_order.T)+t).T
        dis = np.linalg.norm(new_A - B_in_order,axis=1)
        match_num=np.sum(dis<d_thre)
        print("dist is",np.shape(dis),"new_A",np.shape(new_A))

        if (match_num>match_max):
            init_R=R
            init_t=t
            match_max = match_num
    print("best init R ",init_R)
    print("best init t ",init_t)
    #stirng2File(idx1,idx2,init_R,init_t)

 
 

    # NOW, we have the best R.t for the initialization  matrix
    # Do the icp
# %%  



    tgt_data_tree = o3d.geometry.KDTreeFlann(tgt_all_data)
    cost_prev=1000000
    for i in range(20):
        R_,t_,cost_ =find_accociations(A_in_order.T,B_in_order.T,init_R,init_t,0.5,tgt_data_tree)
        init_R = R_
        init_t = t_
        if(cost_>cost_prev):
            break
        else:
            cost_prev=cost_
        
        print("after #",i, " udpate, cost is ",cost_)
    stirng2File(idx1,idx2,init_R,init_t)


 

