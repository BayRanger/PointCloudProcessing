# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def cov(data):
    n= np.shape(data)[0]
    data_mean = np.mean(data,axis=0)
    data = data - data_mean
    return data.transpose()@data/(n-1)

def weight_cov(data,weight,data_mean,n):
    data = data - data_mean
    return weight*data.transpose()@data/(n-1)
    
    

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_center_ = [None]*n_clusters
        self.cluster_cov_ =[None]*n_clusters
        self.k_ = n_clusters
        self.weight_ = None
        self.pi_=np.zeros((n_clusters))
        self.isinited_ = False

    
    '''
    initialize the k mean covariance 
    '''
    
    # 屏蔽开始
    # 更新Var


    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        data= np.array(data)
        print("data shape",np.shape(data))
        if (not self.isinited_):
            self.gmm_init(data)
            self.isinited_ = True
        for time in range(self.max_iter):
            for i in range(self.k_):
                self.weight_[:,i]=self.pi_[i]*multivariate_normal.pdf(data,self.cluster_center_[i],self.cluster_cov_[i])
                #print("cov",self.cluster_cov_[i])
            # 更新W

                #print("weight",self.weight_)
            self.weight_ = self.weight_/np.sum(self.weight_,axis=1,keepdims=True)
            #print("end weight",self.weight_)
            #maximization
            n_k = np.sum(self.weight_,axis=0)
            #print("N_K",n_k)
            # 更新pi
            self.pi_= n_k/self.n_
            #print("pi",self.pi_)
            for i in range(self.k_):
                # 更新Mu
                self.cluster_center_[i]= 1/n_k[i]*np.sum(((self.weight_[:,i])[:,None]*data),axis=0)
                # 更新Var
                self.cluster_cov_[i]=weight_cov(data,self.weight_[:,i],self.cluster_center_[i],n_k[i])
                #print("cluster cov",self.cluster_cov_[i])              
            #print("cluster center",self.cluster_center_)
            #
            #tmp=0
            #for i in range(self.k_):
            #    tmp =tmp+ self.pi_[i]*multivariate_normal.pdf(data,self.cluster_center_[i],self.cluster_cov_[i])
            #tmp=np.sum(np.log(tmp)) 
            #print("benchmark",tmp)   
     

    


        # 屏蔽结束
    
    def predict(self, data):
        idx = np.argmax(self.weight_,axis=1)
        #print("idx",idx)
        return idx

    def gmm_init(self,data):        
        data_max=np.max(data,axis=0)
        data_min=np.min(data,axis=0)
        data=np.array(data)
        print("total data size",np.shape(data))
        self.n_ = np.shape(data)[0]
        #init weight
        self.weight_ = np.zeros((self.n_,self.k_))
        #true_Mu = [[0.5, 1], [5.5, 2.5], [1, 7]]

        for i in range(self.k_):
            #random selection
            self.cluster_center_[i]=data[i]
            #data_min+ (data_max-data_min)/self.k_*i
        self.cluster_center_=np.array(self.cluster_center_)
        r_state = self.closest_centroid(data, self.cluster_center_)
        for i in range(self.k_):
            self.cluster_cov_[i]=np.cov(data[r_state==i].transpose())
            #init p_table
            self.weight_[r_state==i,i]=1
            print("mean",self.cluster_center_[i])
            print("cov",self.cluster_cov_[i])
            print("partial data size:",np.shape(data[r_state==i]))
            self.pi_[i] = np.sum(self.weight_[:,i])/self.n_
            print(i,",prob ",self.pi_[i])
            
        print("weight",self.weight_)
     
    #returns an array containing the index to the nearest centroid for each point
    def closest_centroid(self,points, centroids):
        centroids=np.array(centroids)
        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
       

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
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.gmm_init(X)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

