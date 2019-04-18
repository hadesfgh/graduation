import numpy as np


def pca(X, k):
    sigma = (X.T.dot(X)) / len(X)               # 因为这里的X数据是按行排列
    U, S, V = np.linalg.svd(sigma)              # sigma是n*n矩阵（n指数据维度）
    U = X.dot(U[:, 0:k])                           # U是n*n矩阵,但是只保留前K列，X是n*m矩阵
    return U, S, V                              # 最后返回的U按行排列，此时已经完成降维