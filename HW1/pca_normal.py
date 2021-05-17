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
    # 作业1
    # 屏蔽开始

    data =np.array(data)
    if (correlation==False):
        data_mean = np.mean(data,axis=0)
        data = data - data_mean
    n= np.shape(data)[0]
    c_mat = data.transpose()@data/(n-1)  #correlation or covariance
    #np.cov(c_mat)
    eigenvalues, eigenvectors = np.linalg.eig(c_mat)
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
        # 指定点云路径
    cat_index = 3# 物体编号，范围是0-39，即对应数据集中40个物体/home/chahe/Documents/shenlan_pointcloud/ModelNet40/airplane
    root_dir = '/home/chahe/project/PointCloud3D/dataset/ModelNet40' # 数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.off') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云
    return

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = np.array(v[:,:2]) #点云主方向对应的向量,取前两列
    points_pca = np.array(points)@point_cloud_vector
    print(np.shape(points_pca))
    #points_pca = points_pca.tolist()
    #print(points_pca)
    plt.title("the first two principles of the points")
    plt.plot(points_pca[:,0],points_pca[:,1])
    plt.show()
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    points =np.array(points)
    n=points.shape[0]
    normals = np.zeros((n,3))
    for i in range(n):
        _,idxs,_ =pcd_tree.search_knn_vector_3d(points[i],16)
        points_tmp =points[idxs]
        print(i,", ",points)
        w, v = PCA(points_tmp)
        v_tmp =v[2] 
        normals[i,:]=v[2]
          
    normals = np.array(normals, dtype=np.float64)
    #print(np.shape(normals))
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
