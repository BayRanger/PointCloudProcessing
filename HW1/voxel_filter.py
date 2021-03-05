# -*- coding: UTF-8 -*-
# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    r=leaf_size
    points = np.array(point_cloud)
    print("original point shape ",np.shape(points))
    x_max=np.max(points[:,0])
    x_min=np.min(points[:,0])
    y_max=np.max(points[:,1])
    y_min=np.min(points[:,1])
    z_max=np.max(points[:,2])
    z_min=np.min(points[:,2])
    d_x =np.ceil((x_max-x_min)/r)
    d_y =np.ceil((y_max-y_min)/r)
    d_z =np.ceil((z_max-z_min)/r)
    h_x = np.floor((points[:,0]-x_min)/r)
    h_y = np.floor((points[:,1]-y_min)/r)
    h_z = np.floor((points[:,2]-z_min)/r)
    h = h_x + h_y*d_x +h_z*d_z*d_y
    h=h.reshape(-1,1)
    points_id =np.hstack((points,h))
    points_id =points_id[points_id[:,3].argsort()]
    max_idx =np.max(points_id[:,3])
    min_idx=np.min(points_id[:,3])
    data_size =np.shape(points_id)[0]
    data_dict = {}
    for i in range(data_size):
        if points_id[i,3] not in data_dict.keys():
            data_dict[points_id[i,3]]=points_id[i,:3]
            #use the first element

    val_array=np.array([])
    for ele in data_dict.values():
        val_array=np.append(ele,val_array)
    val_array=val_array.reshape([-1,3]) 
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(val_array, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "/Users/renqian/Downloads/program/cloud_data/11.ply"
    cat_index=5
    root_dir = '/home/chahe/project/shenlan_pointcloud/ModelNet40' # 数据集路径
    cat = os.listdir(root_dir)
    filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.off') # 默认使用第一个点云
    point_cloud_pynt = PyntCloud.from_file(filename)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    #o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 3.0)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
