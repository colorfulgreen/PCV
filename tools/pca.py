from numpy import *
from pylab import *

def pca(X):
    '''主成分分析：
    输入：矩阵 X，其中该矩阵中存储训练数据，每一行为一条训练数据。
    返回：投影矩阵（按照维度的重要性排序）、方差和均值'''

    # 获取维数
    num_data, dim = X.shape
    # 数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA - 使用紧凑技巧
        M = dot(X, X.T)         # 协方差矩阵?
        e, EV = linalg.eigh(M)  # 特征值和特征向量
        tmp = dot(X.T, EV).T    # 这就是紧致技巧?
        V = tmp[::-1]           # 由于最后的特征向量是我们所需要的，所以需要将其逆转
        S = sqrt(e)[::-1]       # 由于特征值是按照递增顺序排序的，所以需要将其逆转
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - 使用 SVD 方法
        U, S, V = linalg.svd(X)
        V = V[:num_data]        # 仅返回前 num_data 维的数据才合理

    return V, S, mean_X



