import numpy as np
import matplotlib.pyplot as plt


def findClosestCentroids(x, centroids):
    idx = []    # 输出由每个样本点距离最近的聚类中心的下标组成的一维数组
    max_dist = 1000000
    for i in range(len(x)):
        minus = x[i]-centroids   # 广播,minus维度与centroids相同
        dist = minus[:, 0]**2+minus[:, 1]**2
        if dist.min() < max_dist:
            ci = np.argmin(dist)  # 样本点距离最近的聚类中心的下标
            idx.append(ci)
    return np.array(idx)


def computeCentroids(x, idx):             # 返回重新计算的聚类中心
    centroids = []
    for i in range(len(np.unique(idx))):  # 返回数组中唯一值排序后的数组
        u_k = x[idx == i].mean(axis=0)    # 找到同一标签的数据，求均值
        centroids.append(u_k)
    return np.array(centroids)


def plotData(x, centroids, idx=None):
    # 可视化数据，并自动分开着色。
    # idx: 最后一次迭代生成的idx向量，存储每个样本分配的簇中心点的值
    # centroids: 包含每次中心点历史记录
    colors = ['gold', 'g', 'gold', 'darkorange', 'salmon', 'olivedrab',
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'gainsboro'
              'coral', 'aliceblue', 'dimgray', 'mintcream','mintcream']
    assert len(centroids[0]) <= len(colors)  # 如果不满足条件报警
    subx = []  # 保存分类后的点
    if idx is not None:
        for i in range(centroids[0].shape[0]):
            x_i = x[idx == i]
            subx.append(x_i)  # subx是二维数组，此时已分完类
    else:
        subx = [x]  # 此时由于没有聚类，直接画出所有的点，不分配颜色
    # 分别画出每个簇的点，并着不同的颜色
    # plt.figure(figsize=(8, 5))
    for i in range(len(subx)):
        xx = subx[i]
        plt.scatter(xx[:, 0], xx[:, 1], c=colors[i],label='Cluster %d'%i)
    # plt.legend(loc='best')      # 显示图例
    # plt.grid(True)              # 生成网格
    # plt.xlabel('x1', fontsize=14)
    # plt.ylabel('x2', fontsize=14)
    # plt.title('Plot of x Points', fontsize=16)
    # 画出簇中心点的移动轨迹
    xx, yy = [], []
    for centroid in centroids:
        xx.append(centroid[:, 0])
        yy.append(centroid[:, 1])
    plt.plot(xx, yy, 'rx--', markersize=8)


def runKmeans(X, centroids, max_iters):   # max_iters是迭代次数
    K = len(centroids)
    centroids_all = []                    # centroids_all保存每次中心点迭代记录，列表类型
    centroids_all.append(centroids)
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx)
        centroids_all.append(centroids)
    return idx, centroids_all