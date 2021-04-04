# 该文件定义了在树中查找数据所需要的数据结构，类似一个中间件

import copy


class DistIndex:
    def __init__(self, distance, index):
        self.distance = distance
        self.index = index

    def __lt__(self, other):
        return self.distance < other.distance



class RadiusNNResultSet:
    def __init__(self, radius):
        self.radius = radius
        self.count = 0
        self.worst_dist = radius
        self.dist_index_list = []
        self.dist_list =[]
        self.index_list=[]
        self.comparison_counter = 0

    def size(self):
        return self.count

    def worstDist(self):
        return self.radius

    def add_point(self, dist, index):
        self.comparison_counter += 1
        if dist > self.radius:
            return

        self.count += 1
        self.dist_index_list.append(DistIndex(dist, index))
        if dist == 0:
            return
        self.dist_list.append(dist)
        self.index_list.append(index)

    def __str__(self):
        self.dist_index_list.sort()
        output = ''
        for i, dist_index in enumerate(self.dist_index_list):
            output += '%d - %.2f\n' % (dist_index.index, dist_index.distance)
        output += 'In total %d neighbors within %f.\nThere are %d comparison operations.' \
                  % (self.count, self.radius, self.comparison_counter)
        return output


