#%%
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import operator


# %%
"""
get the weighted covariance matrix based on the neighborhoods 
"""
def getWeightedCov(pcd_tree,data,this_idx,neigh_idxs,sub_radius):
    center= data[this_idx]
    neigh_pts = data[neigh_idxs]
    n = np.shape(neigh_idxs)[0]
    #print("neight of ",this_idx," is ",n)
    wsum=0
    numerator = np.zeros((3,3))
    for i in range(np.shape(neigh_pts)[0]):
        _,weight,_ =pcd_tree.search_radius_vector_3d(neigh_pts[i],sub_radius)
        wj= 1.0/np.shape(weight)[0]
        wsum+=wj
        vec= np.array([data[i]-center]) 
        tmp =vec.T@vec
        numerator+=wj*tmp/n
    cov=numerator/wsum
    return cov

# %%
"""
The implementation of iss key point selection algorithm
"""
def getIssKeyPointIdx(filename, thre1=0.6,thre2=0.6,r1=2,r2=2,iss_count=500):
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    points = point_cloud_pynt.points
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    points =np.array(points)
    n=points.shape[0]
    normals = np.zeros((n,3))
    lambda_dict={}
    cand_idxs=[]
    iss_idxs=[]
    for i in range(n):
        #search for the idx of neighbors in the data
        _,idxs,_ =pcd_tree.search_radius_vector_3d(points[i],0.5)
        w_cov = getWeightedCov(pcd_tree,points,i,idxs,r1)
        #print("w_cov ", w_cov)
        eigenvalues, eigenvectors = np.linalg.eig(w_cov)
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
        lambda1 = eigenvalues[0]
        lambda2 = eigenvalues[1]
        lambda3 = eigenvalues[2]
        lambda_dict[i] = lambda3
        if (lambda1==0 or lambda2==0 or lambda3 ==0):
            continue
        elif (lambda2**2/lambda1**2)<thre1 and (lambda3**2/lambda2**2<thre2):
            #cand_idx.add(i)
            #prefer to use []
            cand_idxs.append(i)
        #iss_count= 200
    while(len(cand_idxs)>0 and len(iss_idxs)<iss_count):
        max_idx = max(lambda_dict.items(), key=operator.itemgetter(1))[0]
        #print("max_idx",max_idx)
        iss_idxs.append(max_idx)
        del lambda_dict[max_idx]
        _,idxs,_ =pcd_tree.search_radius_vector_3d(points[max_idx],r2)# TODO check threshold
        for ele in idxs:
            try:
                cand_idxs.remove(ele)
                del lambda_dict[ele]
            except:
                pass
    return iss_idxs


if __name__ == "__main__":
    #load data
    cat_index = 16# 物体编号，范围是0-39，即对应数据集中40个物体/home/chahe/Documents/shenlan_pointcloud/ModelNet40/airplane
    root_dir = '/home/chahe/project/PointCloud3D/dataset/ModelNet40' # 数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.off') # 默认使用第一个点云
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
 
    # 从点云中获取点，只对点进行处理
    points = np.array(point_cloud_pynt.points)
    print('total points number is:', points.shape[0])
    #%%

    #pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    iss_idxs = getIssKeyPointIdx(filename)
 
    iss_points = points[iss_idxs]
    print((iss_idxs[353]),iss_idxs[350],iss_idxs[343])
    point_cloud_o3d.points =o3d.utility.Vector3dVector(iss_points)
    o3d.visualization.draw_geometries_with_editing([point_cloud_o3d])