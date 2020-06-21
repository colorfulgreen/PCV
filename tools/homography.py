from numpy import *
from pylab import *

from . import ransac
from importlib import reload
reload(ransac)

def normalize(points):
    '''在齐次坐标下，对点集进行归一化，使最后一行为 1'''
    for row in points:
       row /= points[-1]
    return points


def make_homog(points):
    '''将点集 (dim*n 数组) 转换为齐次坐标表示'''
    return vstack((points, ones((1, points.shape[1]))))


def H_from_points(fp, tp):
    '''使用 DLT 方法，计算单应性矩阵 H，使 fp 映射到 tp。点自动进行归一化。'''
    if fp.shape != tp.shape:
       raise RuntimeError('number of points do not match')

    # 对点进行归一化，使其均值为 0，方差为 1. 因为算法的稳定性
    # 取决于坐标的表示情况和部分数值计算问题，所以归一化操作非常重要。

    # --- 映射起始点 ---
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp = dot(C1, fp)

    # --- 映射对应点 ---
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1/maxstd, 1/maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = dot(C2, fp)

    # 创建用于线性方法的矩阵，对于每个对应对，在矩阵中会出现两行数值
    nbr_correspondences = fp.shape[1]
    A = zeros((2*nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2*i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, tp[0][i]*fp[0][i], tp[0][i]*fp[1][i], tp[0][i]]
        A[2*i+1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1, tp[1][i]*fp[0][i], tp[1][i]*fp[1][i], tp[1][i]]

    # 最小二乘解即为矩阵 SVD 分解后所得矩阵 V 的最后一行
    U, S, V = linalg.svd(A)
    H = V[8].reshape((3,3))

    # 反归一化
    H = dot(linalg.inv(C2), dot(H, C1))

    # 归一化，然后返回 TODO 这里归一化的原因？
    return H / H[2, 2]


def Haffine_from_points(fp, tp):
    '''计算仿射变换 H，使得 tp 是 fp 经过仿射变换 H 得到的'''

    if fp.shape != tp.shape:
       raise RuntimeError('number of points do not match')

    # 对点进行归一化
    # --- 映射起始点 ---
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = dot(C1, fp)
    # print('fp={}\nm={}\nmaxstd={}\nC1={}\nfp_cond={}'.format(fp, m, maxstd, C1, fp_cond))

    # --- 映射对应点 ---
    m = mean(tp[:2], axis=1)
    C2 = C1.copy()    # 两个点集必须进行相同的缩放
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = dot(C2, tp)

    # 因为归一化后点的均值为 0，所以平移量为 0
    A = concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = linalg.svd(A.T)

    # 如Hartley 和 Zisserman 著的 Multiple View Geometry in Computer Vision, Second Edition 所示，
    # 创建矩阵 B 和 C
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = concatenate((dot(C, linalg.pinv(B)), zeros((2,1))), axis=1)
    H = vstack((tmp2, [0,0,1]))

    # 反归一化
    H = dot(linalg.inv(C2), dot(H, C1))

    return H / H[2, 2]


class RansacModel(object):
    '''用于测试单应性矩阵的类，其中单应性矩阵是由 https://scipy-cookbook.readthedocs.io/items/RANSAC.html 上的 ransac.py 计算出来的'''
    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        '''接受由 ransac.py 选择的 4 个对应点，拟合一个单应性矩阵。
           4 个点是计算单应性矩阵所需的最少数目。'''
        # 将其转秩，调用 H_from_points() 计算单应性矩阵
        data = data.T
        fp = data[:3, :4]
        tp = data[:3, :4]
        return H_from_points(fp, tp)

    def get_error(self, data, H):
        '''对所有的对应计算单应性矩阵，然后对每个变换后的点，返回相应的误差'''
        data = data.T
        fp = data[:3]
        tp = data[3:]
        fp_transformed = dot(H, fp)

        # 归一化齐次坐标
        for i in range(3):
            fp_transformed[i] /= fp_transformed[2]

        # 返回每个点的误差
        return sqrt(sum((tp-fp_transformed)**2, axis=0))


def H_from_ransac(fp, tp, model, maxiter=1000, match_threshold=500): # TODO threshold 从书中的 10 加到了 500
    '''使用 RANSAC 稳健性估计点对应间的单应性矩阵 H。
       返回单应性矩阵，和对应该单应性矩阵的正确点对。'''
    # 对应点组
    data = vstack((fp, tp))

    # 计算 H 并返回
    H, ransac_data = ransac.ransac(data.T, model, 4, maxiter, match_threshold, 10, return_all=True, debug=False)
    return H, ransac_data['inliers']
