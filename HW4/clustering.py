# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d 
import random
from pyntcloud import PyntCloud
from numpy.linalg import svd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def check_colinearity(data):
    skew1 = (data[0][1]-data[1][1])/(data[0][0]-data[1][0])
    skew2 = (data[2][1]-data[0][1])/(data[2][0]-data[0][0])
    return abs(skew1-skew2)>0.001

def add_one_column(raw_data):
    data_size = np.shape(raw_data)[0]
    data=np.ones((data_size,4))
    data[:,:3] = raw_data
    return data

def line_param_estiamte(data):
    data=add_one_column(data[:3])
    return svd(data)[-1][-1,:]

def is_inliner(param,threshold,data):
    #try to use broadcast
    data = add_one_column(data)
    result = np.abs(np.array(param)@(data.transpose()))/np.sqrt(np.sum(np.array(param)**2))
    return (result<threshold)
# function test    
def is_inliner_test():
    param=[1,1,1,0]
    data= np.array([[0,0,0],[1,1,1]])
    if((is_inliner(param,1,data)==[True,False]).all()):
        print("is_inliner_test pass")
    else:
        print("is_inliner_test fails")
     

def run_ransac(data,threshold=0.5,max_iterations=400,subset_size=20000):
    #init value
    max_inliner=0
    best_param = None
    #for loop 
    
    data_size= np.shape(data)[0]
    for i in range(max_iterations):
        sample_idx=random.sample(range(data_size),subset_size)
        sub_data= data[sample_idx]
        idxs = sub_data[:3]
        #print("idxs",idxs)
        while(check_colinearity(idxs)==False):
            idxs = [np.random.randint(0,sub_data.shape[0]) for _ in range(3)]
        param= line_param_estiamte(idxs)
        inliner_idx=is_inliner(param,threshold,sub_data)
        inliner_size=np.shape(sub_data[inliner_idx])[0]
        if inliner_size>max_inliner:
            best_param = param
            max_inliner= inliner_size
            print("updated max number",max_inliner)
        
    return best_param
        

        
    #random select point
    #fit the model
    #calulate inliners
    
    #find the best one.

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data,threshold=0.6):
    # 作业1
    # 屏蔽开始
    param = run_ransac(data,threshold)
    inline_indices = is_inliner(param,threshold,data)
    filtered_data =(data[inline_indices==False])
    print("filtered size ",filtered_data.shape)
    return filtered_data


    # 屏蔽结束

    #print('origin data points num:', data.shape[0])
    #print('segmented data points num:', segmengted_cloud.shape[0])
    #return segmengted_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    X = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=0.4, min_samples=8).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_  


    # 屏蔽结束

    return labels

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    root_dir = 'data/' # 数据集路径
    cat = os.listdir(root_dir)
    print("cat",cat)
    #cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)
        #filename="/home/chahe/project/PointCloud3D/PointCloudProcessing/HW4/000000.bin"
        point_cloud_pynt = PyntCloud.from_file(filename)
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
        o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云
        origin_points = read_velodyne_bin(filename)
        segmented_points = ground_segmentation(data=origin_points)
        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index)
def Ransac_test():
    filename="data/000000.bin"
            #point_cloud_pynt = PyntCloud.from_file(filename)
    origin_points = read_velodyne_bin(filename)
    segmented_points = ground_segmentation(data=origin_points)
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(segmented_points)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])
    #print(check_colinearity(([1,1],[2,2],[3,3])))


    #print(np.shape(origin_points))
    #is_inliner_test()
    #test=random.sample(range(100),5)
    #print(test)

    

if __name__ == '__main__':
    #main()
    Ransac_test()


