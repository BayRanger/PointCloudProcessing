# %%
import iss
from pca import *
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import operator
import matplotlib.pyplot as plt
# %%
"""[summary]
"""
def FPFH(data,point_label,tree,radius,bin):
    nei_hist,count =0,0.
    #nei_labels = tree.query_radius(data_with_normal[point_label][0:3].reshape(1,-1),radius)[0]
    _,nei_labels,_ = tree.search_radius_vector_3d(data_with_normal[point_label][0:3].reshape(1,-1),radius)
    nei_labels = np.asarray(list(list(set(nei_labels) - set([point_label]))))  #在邻居点中去除关键点
    data_with_norm = PointWithNorm(data,tree,radius)
    histograms = SPFH(data_with_normal,point_label,tree,radius,bin).astype(np.double)
    for neighbor_label in nei_labels:
        count+=1
    

"""[Simplified Point Feature Histogram]
Compute triplet between query point and its neighbors within r
Ouput is 3 histograms(each has bin bins ) by binning triplet
"""
def SPFH(data_with_normal, point_label, tree, radius, bin):
    alpha,phi,theta = [],[],[] 
    point = data_with_normal[point_label]
    #nei_labels = tree.query_radius(data_with_normal[point_label][0:3].reshape(1,-1),radius)[0]
    print("HHH",np.array(data_with_normal[point_label][0:3]) )
    this_data = np.array(data_with_normal[point_label][0:3])
    #data_with_normal[point_label][0:3].reshape(1,-1)
    _,nei_labels,_ = tree.search_radius_vector_3d(this_data,radius)
    
    nei_labels = np.asarray(list(list(set(nei_labels) - set([point_label]))))  #在邻居点中去除关键点
    while(np.shape(nei_labels)[0]<3):
        radius = radius*2
        _,nei_labels,_ = tree.search_radius_vector_3d(this_data,radius)
        nei_labels = np.asarray(list(list(set(nei_labels) - set([point_label]))))  #在邻居点中去除关键点
    
    local_points = data_with_normal[nei_labels] #得到所有邻居点
    p1 = data_with_normal[point_label][0:3].reshape(1,-1)
    u = data_with_normal[point_label][3:].reshape(1,-1)
    for neighbor in nei_labels:
        p2 = data_with_normal[neighbor][0:3].reshape(1,-1)
        if (np.linalg.norm(p2-p1,2)<0.001):
            continue
        n2 = data_with_normal[neighbor][3:].reshape(1,-1)
        print("neighbor idx ",neighbor,"pt idx ",point_label)
        print("p1:",p1," p2:",p2)
        v = np.cross(u,(p2-p1)/np.linalg.norm(p2-p1,2))
        v= v/np.linalg.norm(v)
        w = np.cross(u,v)
        w=w/np.linalg.norm(w)
        this_alpha = np.dot(v,n2.T)/(np.linalg.norm(n2,2))
        this_phi = np.dot((p2-p1),u.T)/np.linalg.norm(p2-p1,2)
        this_theta = np.arctan2(np.dot(w,n2.T),np.dot(u,n2.T))
        alpha.append(float(this_alpha))
        phi.append(float(this_phi))
        theta.append(float(this_theta))   
        #test the 3 in radians 
        
    #voting
    alpha_hist,_ = np.histogram(np.array(alpha),bins=bin,range=(-1,1),density = True)
    phi_hist,_ = np.histogram(np.array(phi),bins=bin,range=(-1,1),density = True)
    theta_hist,_ = np.histogram(np.array(theta),bins=bin,range=(-np.pi,np.pi),density = True)
    histograms = np.hstack((alpha_hist,phi_hist,theta_hist))
    return histograms 
    

#%%
if __name__ == "__main__":
    cat_index = 16# 物体编号，范围是0-39，即对应数据集中40个物体/home/chahe/Documents/shenlan_pointcloud/ModelNet40/airplane
    root_dir = '/home/chahe/project/PointCloud3D/dataset/ModelNet40' # 数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.off') # 默认使用第一个点云
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d) 
    # 从点云中获取点，只对点进行处理
    points = np.array(point_cloud_pynt.points)
    print('total points number is:', points.shape[0])
    
    # predetermine the key points
    kp_idx1 = 275
    kp_idx2 = 248
    kp_comp = 261
    
    
    
    # %%
    data_with_norm = PointWithNorm(points,pcd_tree,0.5)
    histogram1 = SPFH(data_with_norm,kp_idx1,pcd_tree,2,11)
    histogram2 = SPFH(data_with_norm,kp_idx2,pcd_tree,2,11)
    histogram3 = SPFH(data_with_norm,kp_comp,pcd_tree,2,11)

    print(histogram1)
    print(histogram2)
    print(histogram3)
    plt.plot(histogram1, 'o-', label='similar feature1')
    plt.plot(histogram2, 'o-', label='similar feature2')
    plt.plot(histogram3, 'o-', label='different feature')
    plt.legend()
    plt.title('Histogram of feature')
    plt.show()
    #iss_points = points[iss_idxs]
    #point_cloud_o3d.points =o3d.utility.Vector3dVector(iss_points)
    #o3d.visualization.draw_geometries([point_cloud_o3d])
 # %%
 
