import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import operator
import matplotlib.pyplot as plt

# %%
def VisualizePcd(pcd):
    o3d.visualization.draw_geometries([pcd])

def getOriginPcd(filename):
    """Extract pointcloud information from the raw .bin file

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
    return point_cloud_o3d

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
    print(np.shape(data_np))
    all_data = (np.reshape(data_np, (int(data_np.shape[0]/6), 6)))
    #pcl_data = (all_data[:,:3])
    #print(np.shape(pcl_data))
    point_cloud_o3d.points =  o3d.utility.Vector3dVector(all_data[:,:3])
    point_cloud_o3d.normals =  o3d.utility.Vector3dVector((all_data[:,3:]))

    return point_cloud_o3d

def getnpDatafromFile(filename):
    """Extract numpy array from filename

    Parameters
    ----------
    filename : string
        bin file name

    Returns
    -------
    [numpy.array]
        the numpy array
    """
    pcd_data= getOriginPcd(filename)
    return np.asarray(pcd_data.points)

def getnpDatafromPCD(pointcloud):
    """Acquire numpy data from PointCloud

    Parameters
    ----------
    pointcloud : PontCloud
        point cloud data

    Returns
    -------
    numpy array
        the corresponding numpy data
    """
    return np.asarray(pointcloud.points)


def read_oxford_bin(bin_path):
    '''provided by shenlan
    :param path:
    :return: [x,y,z,nx,ny,nz]: 6xN
    '''
    data_np = np.fromfile(bin_path, dtype=np.float32)
    return np.transpose(np.reshape(data_np, (int(data_np.shape[0]/6), 6)))




if __name__ =='__main__':
    print("test")
    filename1 = "/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/"+"0.bin"
    result = getCuttedPcd(filename1)
    VisualizePcd(result)
    filename2 = "/home/chahe/project/PointCloud3D/dataset/registration_dataset/point_clouds/"+"1.bin"
    result2 = getCuttedPcd(filename2)
    VisualizePcd(result2)   
# %%
