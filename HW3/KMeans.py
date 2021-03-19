# 文件功能： 实现 K-Means 算法

import numpy as np

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.isinited_ = False
        self.cluster_center_=[None]*n_clusters
        
    def closest_centroid(self,points, centroids):
        """returns an array containing the index to the nearest centroid for each point"""
        centroids=np.array(centroids)
        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
        

    def fit(self, data):
        # TODO:
        r_state=np.zeros(data.shape[0])
        # use uniform distribution to initialze the data
        if (not self.isinited_):
            self.center_init(data)
            self.isinited_ = True
        for i in range(self.max_iter_):
            #some presiquite
            if (0):
                break
            #expection
            r_state = self.closest_centroid(data, self.cluster_center_)
            #maximization
            self.maximization(r_state,data)
        return 
            

    def maximization(self,cur_state,data):
        #calculate the cluster mean value
        for i in range(self.k_):
            self.cluster_center_[i]=np.mean(data[cur_state==i],axis=0)
                
    def center_init(self, data):
        data_max=np.max(data,axis=0)
        data_min=np.min(data,axis=0)
        for i in range(self.k_):
            self.cluster_center_[i]=data_min+ (data_max-data_min)/self.k_*i
        self.cluster_center_=np.array(self.cluster_center_)   
        

        
    '''
    function:return the label of each data
    '''
    def predict(self, p_datas):
        return self.closest_centroid(p_datas,self.cluster_center_)

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

