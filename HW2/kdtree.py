# kdtree的具体实现，包括构建和查找

import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet

# Node类，Node是tree的基本组成元素
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output

# 功能：构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
# 输入：
#     key：键
#     value:值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_key_by_vale(key, value):
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted


def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1
def getMaxVarAxis(db):
    db_var_list=np.var(db,axis=0)
    max_axis= np.argmax(db_var_list)
    return max_axis
def getMeanLeftRightIdxSet(db,point_indices, axis):
    db_mean=np.mean(db[point_indices,axis])
    left_idx_set=[]
    right_idx_set=[]
    for i,e in enumerate(db[point_indices, axis]):
        if (e<db_mean):
            left_idx_set.append(i)
        else:
            right_idx_set.append(i)
    return point_indices[left_idx_set],point_indices[right_idx_set],db_mean
    
def getMeanLeftRightIdxSetwithNumpy(db,point_indices, axis):
    db_mean=np.mean(db[point_indices,axis])
    pointidx_larger=point_indices[db[point_indices,axis]>db_mean]
    pointidx_smaller=point_indices[db[point_indices,axis]<=db_mean]
    return pointidx_larger,pointidx_smaller,db_mean


def getMedianLeftRightIdxSet(db,point_indices, axis):
    point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices,axis])
    middle_left_idx = math.ceil(point_indices_sorted.shape[0]/2)-1
    middle_left_point_idx = point_indices_sorted[middle_left_idx]
    middle_left_point_value = db[middle_left_point_idx,axis]
    
    middle_right_idx = middle_left_idx +1
    middle_right_point_idx = point_indices_sorted[middle_right_idx]
    middle_right_point_value=db[middle_right_point_idx,axis] 
    root_value=(middle_left_point_value + middle_right_point_value) * 0.5
    left_idx= point_indices_sorted[0:middle_right_idx]
    right_idx= point_indices_sorted[middle_right_idx:]
    return left_idx, right_idx,root_value
    

# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据
#     point_indices：排序后的键
#     axis: scalar
#     leaf_size: scalar
# 输出：
#     root: 即构建完成的树
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        left_idx,right_idx,root.value = getMedianLeftRightIdxSet(db,point_indices,axis)
        new_axis=axis_round_robin(axis, dim=db.shape[1])
        #new_axis=getMaxVarAxis(db[point_indices,:])
        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           left_idx,
                                           new_axis,
                                           leaf_size)
        root.right = kdtree_recursive_build(root.right,
                                           db,
                                           right_idx,
                                           new_axis,
                                           leaf_size)

    return root


# 功能：翻转一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        pass
        #print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1

# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据
#     leaf_size：scale
# 输出：
#     root：构建完成的kd树
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set：搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 作业2
    # 提示：仍通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:
            kdtree_knn_search(root.left, db, result_set, query)
    if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    # 屏蔽结束

    return False

# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set:搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    
    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    # 屏蔽结束

    return False



def main():
    # configuration
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 1

    db_np = np.random.rand(db_size, dim)
    #print("db is",db_np)
    db_var = np.var(db_np[:, 1])
    #db_var_list=[np.var(db_np[:,i]) for i in db_np.shape()[1]]
    #db_mean=np.mean(db_np[:,1])
    #db_larger=db_np[db_np[:,1]>db_mean]
    db_var_list=np.var(db_np,axis=0)
    max_axis_idx= np.argmax(db_var_list)

    #print("db in one  axis:",np.mean(db_np[:, 1]))
    print("db var",db_var_list)
    print("idx ",max_axis_idx)
    root = kdtree_construction(db_np, leaf_size=leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    query = np.asarray([0, 0, 0])
    result_set = KNNResultSet(capacity=k)
    kdtree_knn_search(root, db_np, result_set, query)
    #
    print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])
    #
    #
    # print("Radius search:")
    # query = np.asarray([0, 0, 0])
    # result_set = RadiusNNResultSet(radius = 0.5)
    # radius_search(root, db_np, result_set, query)
    # print(result_set)


if __name__ == '__main__':
    main()