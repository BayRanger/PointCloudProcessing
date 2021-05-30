import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import operator
import matplotlib.pyplot as plt

# %%

 

"""[extract point cloud inforamtion from raw bin]
"""

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