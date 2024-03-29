import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from fill_normalization import fill, normalization
from PcaByAndrew import pca
from kmeansByAndrew import runKmeans, plotData
from K_means改 import kpp_centers
from problem_data import ProblemData
from predict_data import PredictData
from unknown_data import UnknownData
from test_data_new import TestData


def euler_distance(point1,point2):
    # 计算两点之间的欧拉距离，支持多维
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


def mainpca():
    unknowndata = UnknownData()  # 未知数据产生聚类中心
    unknowndata = fill(unknowndata, 1778, 336, 254)
    unknowndata = normalization(unknowndata, 1778, 336)
    U1, S1, V1 = pca(np.array(unknowndata, dtype='float'), 2)
    init_centroids = np.array(kpp_centers(U1, 2))
    idx, centroids_all = runKmeans(U1, init_centroids, 100)
    centroids = centroids_all[-1]
    print("感知不明数据产生的聚类中心点：\n", centroids[0], centroids[1])
    # plotData(U1, centroids_all, idx)
    # plt.savefig('F:\\导出的图片.png')
    # plt.show()

    problemdata = ProblemData()  # 问题数据
    problemdata = fill(problemdata, 2093, 336, 299)
    problemdata = normalization(problemdata, 2093, 336)
    U2, S2, V2 = pca(np.array(problemdata, dtype='float'), 2)
    # plt.scatter(U2[:, 0], U2[:, 1])

    testdata = TestData()  # 测试正确率的数据
    testdata = fill(testdata, 476, 336, 68)
    testdata = normalization(testdata, 476, 336)
    U3, S3, V3 = pca(np.array(testdata, dtype='float'), 2)
    # plt.scatter(U3[:, 0], U3[:, 1],c='orange')
    # plt.show()

    # plt.subplot2grid((2,2),(0,0))
    # plt.scatter(U2[:, 0], U2[:, 1])
    # plt.subplot2grid((2,2),(0,1))
    # plt.scatter(U3[:, 0], U3[:, 1],c='orange')
    # # plt.savefig('F:\\导出的图片1.png')
    # plt.subplot2grid((2,2),(1,0))
    # plotData(U1, centroids_all, idx)
    # # plt.savefig('F:\\导出的图片3.png')
    # plt.show()

    # a = np.random.randint(0, 84)
    data_7, day, ECI, time, name = PredictData()  # 做感知差识别的数据
    data_7 = fill(data_7, 7, 336, 1)
    data_7 = normalization(data_7, 7, 336)
    U4, S4, V4 = pca(np.array(data_7, dtype='float'), 2)
    data_arg = U4[day]
    print("降维后的感知差识别数据：", data_arg[0], data_arg[1])
    print("ECI:", ECI)
    print("time:", time)
    print("name:", name)

    T = U2  # 预测
    P = U3
    num1 = num2 = 0
    num3 = num4 = 0
    string1 = '该日数据存在感知差问题'
    string2 = '该日数据感知正常'
    for i in range(2093):
        if euler_distance(T[i], centroids[0]) <= euler_distance(T[i], centroids[1]):
            num1 += 1
        else:
            num2 += 1
    if num1 >= num2:
        print("感知差聚类中心点为：", centroids[0][0], centroids[0][1])
        # centroids0为问题小区中心点
        for i in range(476):
            if euler_distance(P[i], centroids[0]) <= euler_distance(P[i], centroids[1]):
                num3 += 1
        print('预测准确度为：', '%.2f' % (100 * num3 / 476), '%')
        dis1 = euler_distance(data_arg, centroids[0])
        dis2 = euler_distance(data_arg, centroids[1])
        if dis1 < dis2:
            string = string1
            print(string)
        else:
            string = string2
            print(string)
    else:
        print("感知正常中心点为:", centroids[1][0], centroids[1][1])
        # centroids1为问题小区中心点
        for i in range(476):
            if euler_distance(P[i], centroids[0]) >= euler_distance(P[i], centroids[1]):
                num4 += 1
        print('预测准确度为：', '%.2f' % (100 * num4 / 476), '%')
        dis1 = euler_distance(data_arg, centroids[0])
        dis2 = euler_distance(data_arg, centroids[1])
        if dis1 > dis2:
            string = string1
            print(string)
        else:
            string = string2
            print(string)
    return string, ECI, time, name


if __name__ == '__main__':
    a, b, c, d = mainpca()
