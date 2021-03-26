## Benchmark

#### 没有改进的 KD Tree

In total 1094 comparison operations.tree depth: 0, max depth: 16

**观察**：KNN的速度远快于brute，但是radius更快
Kdtree: build 133.531, knn 2.734, radius 0.329, brute 8.804

### KD 树所采用的改进方法

#### 添加通过Mean而不是Median对于树进行建立的操作

In total 44 comparison operations.
tree depth: 0, max depth: 16
Kdtree: build 417.527, knn 0.787, radius 0.190, brute 8.488

**观察**： KdTree构造时间相比于naive的方法增加，因为计算Mean的时候没有用numpy，效率偏低。但是comparison的次数明显降低，说明，选择mean划分树时，leaf的分布更加均匀。


#### 添加通过Mean(numpy)对于树进行建立的操作

In total 44 comparison operations.
tree depth: 0, max depth: 16
Kdtree: build 125.107, knn 0.727, radius 0.203, brute 9.379

**观察**： KdTree构造时间相比于naive的方法减小，远低于不用numpy的mean求解 ，体现了numpy计算带来的效率提升，也说明python的for loop非常低效率。

其他参数相比于上一种方法没有明显变化

#### 采用更adaptive的方法选择轴来划分树的结点（with median）

In total 92 comparison operations.
tree depth: 0, max depth: 13
Kdtree: build 259.410, knn 0.884, radius 0.281, brute 8.086

**观察**： KdTree构造时间增加，因为计算方差的时间更长，但是搜索优化，体现在比较次数减少（相比于naive的方法），knn搜索速度加快。同时，radius搜索速度也得到了增加。


#### 采用更adaptive的方法选择轴来划分树的结点（with mean）

In total 36 comparison operations.
tree depth: 0, max depth: 16
Kdtree: build 329.691, knn 0.510, radius 0.124, brute 7.988

**观察**： KdTree构造时间进一步增加，和预期不同，预期是使用numpy的mean会让构建时间减少。但是，knn的速度依然得到了提升，radius也是。

#### OCtree所采用的方法主要按照原本code 要求

Benmark如下

Octree: build 4605.775, knn 0.657, radius 0.491, brute 8.367

**观察**： Octree的构建时间和radius搜索时间明显高于naive的kdtree。Knn时间小于naive的knn，但是和改进过的knn相比，没有明显差距


## 总结

octree的构造时间最长，搜索时间快于naive的kdtree。不过，优化过的kdtree在搜索速度上超过了octree，同时，通过mean而不是meidan来划分树后，效率提升，比较次数减少，说明针对提供的数据，使用mean的分类效果更好。而在通过adaptive的方法选择分类轴后，效率进一步提升。此外，mean的实现用numpy要快于用pythonfor循环。
