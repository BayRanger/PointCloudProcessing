import numpy as np
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet
import matplotlib.pyplot as plt
    

class DBSCAN(object):
    def __init__(self,radius =3, MinPts=2):
        self.radius = radius
        self.min_pts = MinPts
    
    def get_neightbor_ids(self,root, query):
        result_set = RadiusNNResultSet(self.radius)
        kdtree.kdtree_radius_search(root, self.data_, result_set, query)
        nis = result_set.index_list
        return nis
    
    def fit(self,data):
        self.data_ = data
        N =data.shape[0]
        self.labels = -1*np.ones(N)
        visited = np.zeros(N)
        unvisited = list(range(N))
        neightbor_unvisited=[]
        label =-1
        tree_root = kdtree.kdtree_construction(data,leaf_size = 32)
        
        while (len(unvisited)>0):
            ind = unvisited.pop()
            visited[ind]+=1
            n_ids = self.get_neightbor_ids(tree_root,data[ind,:])
            visited[n_ids]+=1
            if len(n_ids) < self.min_pts:
                self.labels[ind] = -1
                continue
            else:
                label +=1
                self.labels[ind]=label
                neightbor_unvisited.extend(n_ids)
                
                while(len(neightbor_unvisited)>0):
                    ind = neightbor_unvisited.pop()
                    visited[n_ids]+=1
                    
                    nn_ids = self.get_neightbor_ids(tree_root,data[ind,:])
                    self.labels[nn_ids]= label
                    if (len(nn_ids)>= self.min_pts):
                        nn_ids = np.array(nn_ids)
                        neightbor_unvisited.extend(nn_ids[visited[nn_ids]>0])
        return
    
    def predict(self,data):
        return self.labels

            
if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 1], [2, 2], [1, 2]]
    #x = generate_X(true_Mu, true_Var)
    #print("this",np.shape(x))
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11],[10,10.1],[10.0,10.0],[10.0,10.8],[7,2],[2,2.1],[2,2.2]])
    #print("that",np.shape(x))

    #print(np.shape(x))
    spec_cls =DBSCAN()
    spec_cls.fit(x)
    cat =spec_cls.predict(x)
    #plt.figure(figsize=(10, 8))
    #plt.axis([-10, 15, -5, 15])
    #print(max(cat))
    #for i in range(int(max(cat))):
    #    plt.scatter(x[cat==i, 0], x[cat==i, 1], s=5)

        

   # plt.show()
    print(cat)
      