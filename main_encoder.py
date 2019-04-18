import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from fill_normalization import fill, normalization
from kmeansByAndrew import runKmeans, plotData
from K_means改 import kpp_centers
from problem_data import ProblemData
from predict_data import PredictData
from unknown_data import UnknownData
from test_data_new import TestData
# from en_coder_4 import encode
from en_coder_5 import encode


def euler_distance(point1, point2):
    # 计算两点之间的欧拉距离，支持多维
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


def mainencoder():
    unknowndata = UnknownData()  # 未知数据产生聚类中心
    unknowndata = fill(unknowndata, 1778, 336, 254)
    unknowndata = normalization(unknowndata, 1778, 336)
    U1 = encode(unknowndata)
    init_centroids = np.array(kpp_centers(U1, 2))
    idx, centroids_all = runKmeans(U1, init_centroids, 100)
    centroids = centroids_all[-1]
    print("感知不明数据产生的聚类中心点：\n", centroids[0], centroids[1])
    # plotData(U1, centroids_all, idx)
    # plt.scatter(U1[:, 0], U1[:, 1])
    # plt.savefig('F:\\导出的图片.png')
    # plt.show()

    tf.reset_default_graph()
    problemdata = ProblemData()  # 问题数据
    problemdata = fill(problemdata, 2093, 336, 299)
    problemdata = normalization(problemdata, 2093, 336)
    U2 = encode(problemdata)
    # plt.scatter(U2[:, 0], U2[:, 1],c='red')

    tf.reset_default_graph()
    testdata = TestData()  # 测试准确度的数据
    testdata = fill(testdata, 476, 336, 68)
    testdata = normalization(testdata, 476, 336)
    U3 = encode(testdata)
    # plt.scatter(U3[:, 0], U3[:, 1],c='blue')
    # plt.show()

    # plt.subplot2grid((2,2),(0,0))
    # plt.scatter(U2[:, 0], U2[:, 1],c='red')
    # plt.subplot2grid((2,2),(0,1))
    # plt.scatter(U3[:, 0], U3[:, 1],c='blue')
    # # plt.savefig('F:\\导出的图片1.png')
    # plt.subplot2grid((2,2),(1,0))
    # # plt.scatter(U1[:, 0], U1[:, 1],c='gold')
    # plotData(U1, centroids_all, idx)
    # # plt.savefig('F:\\导出的图片3.png')
    # plt.show()

    tf.reset_default_graph()
    # a = np.random.randint(0, 84)
    data_7, day, ECI, time, name = PredictData()  # 做感知差识别的数据
    data_7 = fill(data_7, 7, 336, 1)
    data_7 = normalization(data_7, 7, 336)
    U4 = encode(data_7)
    data_arg = U4[day]
    print("降维后的感知差识别数据：", data_arg[0], data_arg[1])
    print("ECI:", ECI)
    print("time:", time)
    print("name:", name)

    # plt.show()

    T = U2
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
        print("感知差数据聚类中心点为：", centroids[0][0], centroids[0][1])
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
        print("感知正常聚类中心点为:", centroids[1][0], centroids[1][1])
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
    a, b, c, d = mainencoder()



