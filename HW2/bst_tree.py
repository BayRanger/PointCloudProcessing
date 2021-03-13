import numpy as np
import math
import copy

#nide definition
class Node:
    def __init__(self, key, value =-1):
        self.left = None
        self.right = None
        self.key = key
        self.value = value
    
    def __str__(self):
        return "key: %s, value: %s" % (str(self.key), str(self.value))

        
#node construction

def insert(root, key, value=-1):
    if root is None:
        root = Node(key,value)
    else:
        if key < root.key:
            root.left = insert(root.left, key, value)
        elif key>root.key:
            root.right = insert(root.right, key, value)
        #don't insert if key already exists in the tree
        else:
            pass
    return root

#root = None
#for i, point in enumerate(data):
    #value in the node is the index of a point in the array
    #root =insert(root,point,i)
    
#search recursively
def search_recursive(root,key):
    if root is None or root.key ==key:
        return root
    if key<root.key:
        return search_recursive(root.left,key)
    elif key > root.key:
        return search_recursive(root.right,key)


#search Iteratively
def search_iterative(root,key):
    current_node = root
    while current_node is not None:
        if current_node.key==key:
            return current_node
        elif key<current_node.key:
            current_node = current_node.left
        elif key>current_node.key:
            current_node = current_node.right
        return current_node

'''
 Depth First Traversal

* **Algorithm Inorder(tree)**
   1. Traverse the left subtree, i.e., call Inorder(left-subtree)
   2. Visit the root.
   3. Traverse the right subtree, i.e., call Inorder(right-subtree)
   
* **Algorithm Preorder(tree)**
   1. Visit the root.
   2. Traverse the left subtree, i.e., call Preorder(left-subtree)
   3. Traverse the right subtree, i.e., call Preorder(right-subtree)
   
* **Algorithm Postorder(tree)**
   1. Traverse the left subtree, i.e., call Postorder(left-subtree)
   2. Traverse the right subtree, i.e., call Postorder(right-subtree)
   3. Visit the root.
'''
def inorder(root):
    if root is not None:
        inorder(root.left)
        print(root)
        inorder(root.right)

def preorder(root):
    if root is not None:
        print(root)
        preorder(root.left)
        preorder(root.right)
        
def postorder(root):
    if root is not None:
        postorder(root.left)
        postorder(root.right)
        print(root)

class KNNResultSet:
    def __init__(self, capacity):
        self.capacity = capacity
        self.count = 0
        self.worst_dist = 1e10
        self.dist_index_list = []
        for i in range(capacity):
            self.dist_index_list.append(DistIndex(self.worst_dist, 0))

        self.comparison_counter = 0

    def size(self):
        return self.count

    def full(self):
        return self.count == self.capacity

    def worstDist(self):
        return self.worst_dist

    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.worst_dist:
            return

        if self.count < self.capacity:
            self.count += 1

        i = self.count - 1
        while i > 0:
            if self.dist_index_list[i-1].distance > dist:
                self.dist_index_list[i] = copy.deepcopy(self.dist_index_list[i-1])
                i -= 1
            else:
                break

        self.dist_index_list[i].distance = dist
        self.dist_index_list[i].index = index
        self.worst_dist = self.dist_index_list[self.capacity-1].distance
        
    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d comparison operations.' % self.comparison_counter
        return output
class DistIndex:
    def __init__(self,distance, index):
        self.distance = distance
        self.index = index
    def __lt__(self,other):
        return self.distance <other.distance
    
def knn_search(root: Node, result_set: KNNResultSet, key):
    if root is None:
        return False

    # compare the root itself
    result_set.add_point(math.fabs(root.key - key), root.value)
    if result_set.worstDist() == 0:
        return True

    if root.key >= key:
        # iterate left branch first
        if knn_search(root.left, result_set, key):
            return True
        elif math.fabs(root.key-key) < result_set.worstDist():
            return knn_search(root.right, result_set, key)
        return False
    else:
        # iterate right branch first
        if knn_search(root.right, result_set, key):
            return True
        elif math.fabs(root.key-key) < result_set.worstDist():
            return knn_search(root.left, result_set, key)
        return False

    
db_size =8
k =5
radius =2.0
data = np.random.permutation(db_size).tolist()
data=[3, 4, 7, 0, 6, 5, 1, 2]
print("data",data)
root = None
for i,point in enumerate(data):
    root =insert(root,point,i)
query_key =6
result_set =KNNResultSet(capacity=k)
knn_search(root,result_set,query_key)
print("kNN search")
print("index - distance")
print(result_set)

#print(result_set.comparison_counter)
#print(result_set.worst_dist)# = Ra