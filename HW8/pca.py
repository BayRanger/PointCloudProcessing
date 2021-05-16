#implement the calculation of PCA and the norm

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量

def PCA(data, correlation=False, sort=True):
    data =np.array(data)
    if (correlation==False):
        data_mean = np.mean(data,axis=0)
        data = data - data_mean
    n= np.shape(data)[0]
    c_mat = data.transpose()@data/(n-1)  #correlation or covariance
    #np.cov(c_mat)
    eigenvalues, eigenvectors = np.linalg.eig(c_mat)
 
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def PointNorm(points,pcd_tree,radius):
    points =np.array(points)
    n=points.shape[0]
    normals = np.zeros((n,3))
    for i in range(n):
        _,idxs,_ =pcd_tree.search_radius_vector_3d(points[i],radius)
        points_tmp =points[idxs]
        while(np.shape(points_tmp)[0]<3):
            radius = radius*2
            _,idxs,_ =pcd_tree.search_radius_vector_3d(points[i],radius)
            points_tmp =points[idxs]
            
        #print(i,", ",np.shape(points_tmp))
        _, v = PCA(points_tmp)
        v_tmp =v[2] 
        normals[i,:]=v[2]
          
    normals = np.array(normals, dtype=np.float64)
    return normals

def PointWithNorm(points,pcd_tree,radius):
    normals = PointNorm(points,pcd_tree,0.1)
    data_with_norm = np.concatenate((points,normals),axis=1)
    return data_with_norm