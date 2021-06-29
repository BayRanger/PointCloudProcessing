import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

fx= 1086.160899
fy= 1090.242963
cx= 940.067502
cy = 586.740077

k_data=[fx , 0, cx, 0,fy,cy, 0,0, 1]
k_mat =np.array(k_data).reshape((3,3))
rt_data =[9.9934608718980233e-01, -1.5715484428488590e-02,-3.2564114721728155e-02, -1.7434527332030676e-02,
           3.2359037356803094e-02, -1.3131917124154624e-02,9.9939003669937865e-01, 1.7171139776467173e-01,
           -1.6133527815482926e-02, -9.9979026615676858e-01,-1.2614792047622947e-02, -4.5251036141047592e-02, 
           0., 0., 0., 1.]
rt_data = np.array(rt_data).reshape((4,4))

def invT(T):
    R=T[:3,:3]
    t=T[:3,3]
    new_T=np.eye(4)
    new_T[:3,:3]=R.T
    new_T[:3,3] = -R.T@t
    return new_T

T_inv = invT(rt_data)
R_data = T_inv[:3,:3]
# init_point =np.array([0,1,0])
# print(R_data@init_point)
def transform_data(pointdata,T):
    R=T_inv[:3,:3]
    t=T_inv[:3,3].reshape((3,1))
    #r_lidar_data =[-1,0,0,0,-1,0,0,0,1]#z
    #r_lidar_data =[-1,0,0,0,1,0,0,0,1] #y
    #r_lidar_data =[1,0,0,0,1,0,0,0,1] #x

    #R_lidar = np.array(r_lidar_data).reshape((3,3))
    pointdata = R@pointdata.T +t
    
    return pointdata.T
    

    
pcl_data=[]
filename="/home/chahe/project/theis_ws/prototype/pcl_process/build/one_frame.txt"
with open(filename, 'r') as f:
    for line in f:
        line =line.strip().split(",")
        line = list(map(float,line))
        pcl_data.append(line)
raw_data = np.array(pcl_data)
data = transform_data(raw_data,rt_data)
data =data[data[:,2]>0]
#data =data[data[:,2]<20]

#print(data)
pc_view = o3d.geometry.PointCloud()
pc_view.points = o3d.utility.Vector3dVector(data)
#o3d.visualization.draw_geometries([pc_view])
#o3d.visualization.draw_geometries([data])





#process data with k
data =k_mat@data.T
print(np.shape(data[:2,:]),np.shape(data[2,:]))

camera_data =data[:2,:]/data[2,:]
print(camera_data)
plt.plot(camera_data[0,:],camera_data[1,:],'o')
plt.xlim(-1000, 3000);
plt.ylim(-1000, 2000);

plt.show()