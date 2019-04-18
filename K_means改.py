import math
import random
import numpy as np


def euler_distance(point1, point2):
    # 计算两点之间的欧拉距离，支持多维
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def kpp_centers(data_set, k):
    # 从数据集中返回 k 个对象可作为质心
    cluster_centers = []
    cluster_centers.append(random.choice(data_set))  # 步骤一：随机选取一个样本作为第一个聚类中心
    d = [0 for _ in range(len(data_set))]  # d是len个0组成的列表
    totle = 0
    id = -1
    for _ in range(1, k):
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers)  # 步骤二：计算每个样本与当前已有类聚中心最短距离
            d[i] = math.pow(d[i], 2)
            totle += d[i]
        dd = np.array(d)/totle  # 转换为ndarray处理
        randValue = np.random.rand(1)
        cum = dd.cumsum()
        id = np.searchsorted(cum, randValue, sorter=np.argsort(cum))  # 步骤三:轮盘法选出下一个聚类中心
        # searchsorted(b,a)查找a在b中的位置，返回位置索引值
        cluster_centers.append(data_set[int(id)])
        totle=0
    return cluster_centers  # 步骤四：选出 k 个聚类中心
