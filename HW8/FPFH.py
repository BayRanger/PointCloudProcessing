import iss
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import operator



if __name__ == "__main__":
    print ("fuck")
    cat_index = 29# 物体编号，范围是0-39，即对应数据集中40个物体/home/chahe/Documents/shenlan_pointcloud/ModelNet40/airplane
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
    iss_idxs = iss.getIssKeyPointIdx(filename)
 
    iss_points = points[iss_idxs]
    point_cloud_o3d.points =o3d.utility.Vector3dVector(iss_points)
    o3d.visualization.draw_geometries([point_cloud_o3d])