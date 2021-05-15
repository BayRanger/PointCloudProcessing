#implement the calculation of PCA and the norm

#%%
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import operator

# %%
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

def main():
    #%% 
    #load data
    cat_index = 29# 物体编号，范围是0-39，即对应数据集中40个物体/home/chahe/Documents/shenlan_pointcloud/ModelNet40/airplane
    root_dir = '/home/chahe/project/PointCloud3D/dataset/ModelNet40' # 数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.off') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    #o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])
    #%%

    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    points =np.array(points)
    n=points.shape[0]
    normals = np.zeros((n,3))
    lambda_dict={}
    #cand_idx=set()
    cand_idxs=[]
    iss_idxs=[]

    thre1=0.6
    thre2=0.6
    # r1 is the range to count weighted covariance matrix
    # r2 is usedc to define the NMS
    #iss_count
    r1=2
    r2=2
    for i in range(n):
        #search for the idx of neighbors in the data
        _,idxs,_ =pcd_tree.search_radius_vector_3d(points[i],0.5)

            
        w_cov = getWeightedCov(pcd_tree,points,i,idxs,r1)
        #print("w_cov ", w_cov)
        eigenvalues, eigenvectors = np.linalg.eig(w_cov)
    # 屏蔽结束
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
        lambda1 = eigenvalues[0]
        lambda2 = eigenvalues[1]
        lambda3 = eigenvalues[2]
        lambda_dict[i] = lambda3
        #print(lambda1,", ",lambda2,", ",lambda3)
    # decide whether to keep points with
        if (lambda1==0 or lambda2==0 or lambda3 ==0):
            continue
        elif (lambda2**2/lambda1**2)<thre1 and (lambda3**2/lambda2**2<thre2):
            #cand_idx.add(i)
            #prefer to use []
            cand_idxs.append(i)
            #print("add ",i)
    # %%
    #NMS
    # max_idx = max(lambda_dict.items(), key=operator.itemgetter(1))[0]
    # iss_idxs.append(max_idx)
    iss_count= 200
    while(len(cand_idxs)>0 and len(iss_idxs)<iss_count):
        max_idx = max(lambda_dict.items(), key=operator.itemgetter(1))[0]
        print("max_idx",max_idx)
        iss_idxs.append(max_idx)
        del lambda_dict[max_idx]
        # delete the elements in the range of r2
        _,idxs,_ =pcd_tree.search_radius_vector_3d(points[max_idx],r2)# TODO check threshold
        for ele in idxs:
            try:
                cand_idxs.remove(ele)
                del lambda_dict[ele]
            except:
                pass
    
                
                
            
    # %% visilize
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)

    pcl_points = points[iss_idxs]
    point_cloud_o3d.points =o3d.utility.Vector3dVector(pcl_points)
    o3d.visualization.draw_geometries([point_cloud_o3d])
#%%
if __name__ == '__main__':
    main()

 
