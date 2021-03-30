"""[summary]
this file is to implement the spectral clustering
"""

from numpy import *
import pylab
import random,math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from KMeans import K_Means
from scipy.spatial import KDTree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet

def get_dist(pointA,pointB):
    return(np.linalg.norm(np.array(pointA)-np.array(pointB)))

def gauss(x,sigma=1):
    return np.exp(-x*x/(2*sigma*sigma))
    
def get_dist_array(pointA,pointB):
    return(np.linalg.norm(np.array(pointA)-np.array(pointB),axis=1))
class spec_cluster(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.simi_graph=None
        self.D_mat=None
        self.lap_mat=None
        self.normed_lap_mat=None
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.isinited_ = False
        self.cluster_center_=[None]*n_clusters
        
    def fit(self,data,method="radius"):
        data_num = np.shape(data)[0]
        print("number of data ",data_num)
        self.simi_graph=np.zeros((data_num,data_num))
        if method=="fully_connect":
            for i in range(np.shape(data)[0]):
                self.simi_graph[i] = get_dist_array(data,data[i])
            #self.simi_graph=np.max(self.simi_graph)*2- self.simi_graph
            self.simi_graph= gauss(self.simi_graph)
            #for i in range(np.shape(data)[0]):
            #    self.simi_graph[i,i]=0
            #print(self.simi_graph)
        elif (method=="radius"):
            root = kdtree.kdtree_construction(data, 16)
            for di, datum in enumerate(data):
                #print(datum)
                result_set = RadiusNNResultSet(radius=10)
                kdtree.kdtree_radius_search(root, data, result_set, datum)
                nis = result_set.index_list
                ndists = result_set.dist_list
                #print("nis",nis)
                #print("ndists",ndists)
                for ni,ndist in zip(nis,ndists):
                    self.simi_graph[di][ni]=self.simi_graph[ni][di]=gauss(ndist,0.2)
                #print("graph",self.simi_graph)
                

                        
        elif (method=="knn"):
            print("knn is implemented")
            tree=KDTree(data)
            for di, datum in enumerate(data):
                ndists,nis = tree.query([datum],20)
                nis=nis[0]
                ndists = ndists[0]
                for ni,ndist in zip(nis,ndists):
                    if ni==di: continue
                    #print("HHHHHH",ni,di,ndist)
                    self.simi_graph[di][ni]=self.simi_graph[ni][di]=gauss(ndist)
            
            
        else:
            print("not available")
        #print(self.simi_graph)
        
    def predict(self,data):
        #第i个点连出的所有线的和
        self.D_mat = np.diag(np.sum(self.simi_graph,axis=1))
        #print(self.D_mat,self.simi_graph)
        self.lap_mat = (self.D_mat - self.simi_graph)
        #self.lap_mat=np.linalg.inv(self.D_mat)*self.simi_graph
        eigenvalues, eigenvectors = np.linalg.eig(self.lap_mat)
        sort = eigenvalues.argsort()
        eigenvalues = eigenvalues[sort]
        #print("eigenvalues",eigenvalues)
        #print("eigenvalue",eigenvalues)
        eigenvectors = eigenvectors[:, sort]
        k_eigenvectors=eigenvectors[:,:self.k_]
        #print("eigen vectors", eigenvectors)
        print("k eigen shape",np.shape(k_eigenvectors))
        k_means = K_Means(self.k_)
        #print("eigen shape",np.shape(k_eigenvectors))
        k_means.fit(k_eigenvectors)
        cat = k_means.predict(k_eigenvectors)
        return cat
        

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
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
    true_Var = [[1, 3], [2, 2], [6, 2]]
    #x = generate_X(true_Mu, true_Var)
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

    #print(np.shape(x))
    spec_cls =spec_cluster(2)
    spec_cls.fit(x)
    cat =spec_cls.predict(x)
    print(cat)


    #cat = gmm.predict(X)
    #print(cat)
    # 初始化