import argparse
import glob
import os
from numpy.core.fromnumeric import shape
from numpy.lib.type_check import imag
import shutil
import numpy as np
import struct
import pandas as pd
import open3d as o3d
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import time
import progressbar
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from multiprocessing import cpu_count
import time


mutex = Lock()
def write_index(index):
    time.sleep(.5)
    mutex.acquire()
    filename= os.path.join(output_dir,f'{index:06d}.txt')
    file = open(filename, "w") 
    file.write(("Your text goes here, # %d"%index)) 
    file.close()
    mutex.release()
    return 
os.chdir(r"/home/chahe/project/PointCloud3D/PointCloudProcessing/Project")

output_dir = "thread_test"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)
index=0
data=(np.arange(10))
start = time.time()
# for i in range(10):
#     write_index(i)
with ThreadPoolExecutor(16) as ex:
    ex.map(write_index, data)
end=time.time()
print("The time cost is ", end-start)
    
"""
目前的理解,线程池的底部是queue
"""

#print(dir(mutex))



