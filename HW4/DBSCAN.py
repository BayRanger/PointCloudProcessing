"""[summary]
this file is to implement the DBSCAN algorithm
"""

from numpy import *
import pylab
import random,math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from scipy.spatial import KDTree
import kdtree as kdtree
from result_set import RadiusNNResultSet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

class dbscan(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self,min_sample= 4, radius =1):
 
        self.isinited_ = False
        self.data_status_ = None  # data_size, if visited
        self.min_samples= min_sample
        self.search_radius = radius
        
    def cluster_init(self,data):
        self.data_num_ = np.shape(data)[0]
        self.data_status_= np.array([True]*self.data_num_)# every data point is not visisted at the beginning,true means unvisited
        self.data_label_=np.zeros(self.data_num_)
        self.data_= data
        self.root_ = kdtree.kdtree_construction(data, 16)

        
        
 #input: the idx list which shoule be visited
 # the cluster idx which it belongs to
 
    def check_neighbor(self,idx,cluster_idx,layer):
        #if it is visited already
        radius=self.search_radius
        self.data_status_[idx]=False
        result_set = RadiusNNResultSet(radius)
        #print(np.shape(self.data_[idx]),(self.data_[idx]))
        kdtree.kdtree_radius_search(self.root_, self.data_, result_set, self.data_[idx])
        nis = result_set.index_list
        ndists = result_set.dist_list
        #print("idx ",idx," nis",nis," ndists",ndists)
        true_nis=np.count_nonzero(self.data_status_[nis])
        if (true_nis>self.min_samples):
            layer+=1
            #print("set idx ",idx,"as label",cluster_idx)
            self.data_label_[idx]=cluster_idx
            for sub_idx in nis:
                #print("nis" ,nis)
                if self.data_status_[sub_idx]==True:
                    #print("set idx inner ",sub_idx,"as label",cluster_idx)        
                    self.data_label_[sub_idx]=cluster_idx
                    self.check_neighbor(sub_idx,cluster_idx,layer)
                    #print("iter layer ",layer)

            return layer
        else:
            #print("layer ",layer)
            return layer
 
        
    def cluster_search(self,cls_idx):
        #print("cls idx:",cls_idx)
        choice_idx=random.choice(arange(self.data_num_)[self.data_status_])
        if(self.check_neighbor(choice_idx,cls_idx,layer=0)>0):
            #print("it is true")
            return True
        #else:
            #the data label should be corrected to a invalid value
            #print("set label ",choice_idx," as ",-1)

            #self.data_label_[choice_idx]=-1
            #return False
            
        
    def fit(self,data):
        self.cluster_init(data)
        cls_idx= 1
        while(len(data[self.data_status_])>0):
            if(self.cluster_search(cls_idx)):
                cls_idx+=1
                #print("cls +1")
                
            
    def predict(self,data):
        # we set noise as -1
        return self.data_label_
        
# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 40, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 60, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 100, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 1], [2, 2], [1, 2]]
    x = generate_X(true_Mu, true_Var)
    #print("this",np.shape(x))
    #x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],[10,10.1],[10.0,10.0],[10.0,10.8],[7,2],[2,2.1],[2,2.2]])
    #print("that",np.shape(x))

    #print(np.shape(x))
    spec_cls =dbscan()
    spec_cls.fit(x)
    labels =spec_cls.predict(x)
    #X = StandardScaler().fit_transform(x)

# #############################################################################
# Compute DBSCAN
    '''
    db = DBSCAN(eps=0.3, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print("labels",labels)
    '''
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    print(max(labels))
    for i in range(-1,int(max(labels))):
        plt.scatter(x[labels==i, 0], x[labels==i, 1], s=5)

        

    plt.show()
    #print(cat)


    #cat = gmm.predict(X)
    #print(cat)
    # 初始化