# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct
from scipy.spatial import KDTree

import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def KDTreeBenchmark(root_dir,files,k,leaf_size,radius,feature=None,feature2=None):
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    iteration_num = 0
    for file in files:
        if file.find('bin') == -1:
            continue
        iteration_num += 1
        filename = os.path.join(root_dir, file)
        db_np = read_velodyne_bin(filename)

        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size,feature,feature2)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        print("result set from KD Tree\n",result_set)
        knn_time_sum += time.time() - begin_t
        # print("--------")
        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        #print(result_set)
        radius_time_sum += time.time() - begin_t
        #print("--------")
        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        #print(nn_idx[0:k])
        #print(nn_dist[0:k])
        brute_time_sum += time.time() - begin_t
        depth = [0]
        max_depth = [0]
        kdtree.traverse_kdtree(root, depth, max_depth)
        print("tree depth: %d, max depth: %d" % (depth[0],max_depth[0]))
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                        knn_time_sum * 1000 / iteration_num,
                                                                        radius_time_sum * 1000 / iteration_num,
                                                                        brute_time_sum * 1000 / iteration_num))

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1.0

    root_dir = './' # 数据集路径
    files = os.listdir(root_dir)

    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    iteration_num = 0
    for file in files:
        if file.find('bin') == -1:
            continue
        iteration_num += 1
        filename = os.path.join(root_dir, file)
        db_np = read_velodyne_bin(filename)
        #point_indices=np.array([1,2,3,4,5,6,7])
        #db_larger=point_indices[db_np[point_indices,1]>0.4]
        #db_larger_idx=np.where
        #print("test data",db_np[point_indices,1])
        #print("test index",db_np[point_indices,1]>0.4)
        #print("test1",db_larger,db_np[db_larger,1])

        begin_t = time.time()
        root = octree.octree_construction(db_np, leaf_size, min_extent)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        octree.octree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search_fast(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
                                                                     knn_time_sum*1000/iteration_num,
                                                                     radius_time_sum*1000/iteration_num,
                                                                     brute_time_sum*1000/iteration_num))

    print("--------------")
    KDTreeBenchmark(root_dir,files,k,leaf_size,radius)
    print("--------------")
    KDTreeBenchmark(root_dir,files,k,leaf_size,radius,"mean")
    print("--------------")
    KDTreeBenchmark(root_dir,files,k,leaf_size,radius,"meanraw")

    print("--------------")
    KDTreeBenchmark(root_dir,files,k,leaf_size,radius,"median","adapt")
    print("--------------")
    KDTreeBenchmark(root_dir,files,k,leaf_size,radius,"meanraw","adapt")
    print("--------------")
    KDTreeBenchmark(root_dir,files,k,leaf_size,radius,"mean","adapt")

    # construction_time_sum = 0
    # knn_time_sum = 0
    # radius_time_sum = 0
    # brute_time_sum = 0
    # iteration_num = 0
    # for file in files:
    #     if file.find('bin') == -1:
    #         continue
    #     iteration_num += 1
    #     filename = os.path.join(root_dir, file)
    #     db_np = read_velodyne_bin(filename)

    #     begin_t = time.time()
    #     root = kdtree.kdtree_construction(db_np, leaf_size)
    #     construction_time_sum += time.time() - begin_t

    #     query = db_np[0,:]

    #     begin_t = time.time()
    #     result_set = KNNResultSet(capacity=k)
    #     kdtree.kdtree_knn_search(root, db_np, result_set, query)
    #     print("result set from KD Tree\n",result_set)
    #     knn_time_sum += time.time() - begin_t
    #     # print("--------")
    #     begin_t = time.time()
    #     result_set = RadiusNNResultSet(radius=radius)
    #     kdtree.kdtree_radius_search(root, db_np, result_set, query)
    #     #print(result_set)
    #     radius_time_sum += time.time() - begin_t
    #     #print("--------")
    #     begin_t = time.time()
    #     diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    #     nn_idx = np.argsort(diff)
    #     nn_dist = diff[nn_idx]
    #     #print(nn_idx[0:k])
    #     #print(nn_dist[0:k])
    #     brute_time_sum += time.time() - begin_t
    #     depth = [0]
    #     max_depth = [0]
    #     kdtree.traverse_kdtree(root, depth, max_depth)
    #     print("tree depth: %d, max depth: %d" % (depth[0],max_depth[0]))
    # print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
    #                                                                     knn_time_sum * 1000 / iteration_num,
    #                                                                     radius_time_sum * 1000 / iteration_num,
    #                                                                     brute_time_sum * 1000 / iteration_num))

    print("--------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    iteration_num = 0
    for file in files:
        if file.find('bin') == -1:
            continue
        iteration_num += 1
        filename = os.path.join(root_dir, file)
        db_np = read_velodyne_bin(filename)

        begin_t = time.time()
        tree = KDTree(db_np)
        construction_time_sum += time.time() - begin_t

        query = db_np[0,:]

        begin_t = time.time()
        result_set = tree.query(([query[0], query[1], query[2]]), k=k, p=2)
        print("result set",result_set)
        knn_time_sum += time.time() - begin_t
        # print("--------")
        begin_t = time.time()
        result_set = tree.query_ball_point(([query[0], query[1], query[2]]), radius)
        # print(result_set)
        radius_time_sum += time.time() - begin_t
        # print("--------")
        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
        print("scipy idx",nn_idx[0:k])
        print("scipy dist",nn_dist[0:k])
    print("Scipy Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))


if __name__ == '__main__':
    main()