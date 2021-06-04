# %%
import numpy as np
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import operator
import matplotlib.pyplot as plt


def parse_bin_to_pcd(filename):
    """Extract pointcloud information from the raw .bin file
    and delete the norm information

    Parameters
    ----------
    filename : [string]
        full path of the bin file

    Returns
    -------
    PointCloud
        the poincloud information
    """
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    data_np = np.fromfile(filename, dtype=np.float32)
    #data_np =  np.asarray(point_cloud_o3d.points)
    #print(np.shape(data_np))
    all_data = (np.reshape(data_np, (int(data_np.shape[0]/6), 6)))

    point_cloud_o3d.points =  o3d.utility.Vector3dVector(all_data[:,0:3])
    point_cloud_o3d.normals =  o3d.utility.Vector3dVector((all_data[:,3:6]))

    return point_cloud_o3d

def visualize_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])
    
    
    # %%
if __name__ == "__main__":
    
#step 1 数据读取
    #load data
    idx1="1"
    idx2="456"
    filename1 = ("/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/{}.bin").format(idx1)
    filename2 = "/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/"+idx2+".bin"
    src_pcd = parse_bin_to_pcd(filename1)
    tgt_pcd = parse_bin_to_pcd(filename1)

    #visualize_pcd(source_pcd)
    #could not visualize now!
# %%