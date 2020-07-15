'''使用八点法计算基础矩阵 F'''

from numpy import *
from pylab import *


class RansacModel(object):
    '''用 RANSAC 计算基础矩阵'''

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        '''使用选择的 8 对对应点计算基础矩阵'''
        # 转置，并将数据分成两个点集
        data = data.T
        x1 = data[:3, :8]
        x2 = data[3:, :8]

        # 估计基础矩阵，并返回
        F = compute_fundamental_normalized(x1, x2)
        return F

    def get_error(self, data, F):
        '''计算所有对应的 x^T F x，并返回每个变换后点的误差'''
        # 转置，并将数据分成两个点集
        data = data.T
        x1 = data[:3]
        x2 = data[3:]

        # 将 Sampson 距离用作误差度量
        Fx1 = dot(F, x1)
        Fx2 = dot(F, x2)
        denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        err = (diag(dot(x1.T, dot(F,x2))))**2 / denom

        # 返回每个点的误差
        return err

def compute_fundamental(x1, x2):
    '''使用归一化(即w1=w2=1)的八点算法，从对应点 (x1, x2 3*n) 的数组中计算基础矩阵
    每行由如下构成：[x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    '''
    n = x1.shape[1]
    if x2.shape[1] != n:
       raise ValueError("Number of points don't match.")

    # 创建方程对应的矩阵
    A = zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i]]

    # 计算线性最小二乘解
    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # TODO 限制 F
    # 通过将最后一个奇异值置 0，使秩为 2
    U, S, V = linalg.svd(F)
    S[2] = 0
    F = dot(U, dot(diag(S), V))

    return F


def compute_epipole(F):
    '''从基础矩阵 F 中计算右极点（可以使用 F.T 获得左极点）'''

    # 返回 F 的零空间（Fx=0）
    U, S, V = linalg.svd(F)
    e = V[-1]
    return e/e[2]


def plot_epipolar_line(im, F, x, epipole=None, show_epipole=True):
    '''在一幅图像中，绘制外极点和外极线 F*x=0. F 是基础矩阵，x 是另一幅图像中的点.'''
    m, n = im.shape[:2]
    line = dot(F, x) # TODO 外极线为什么是 Fx=0 ?

    # 外极线参数和值
    t = linspace(0, n, 100)
    lt = array([(line[2]+line[0]*tt) / (-line[1]) for tt in t])

    # 仅仅处理位于图像内部的点和线
    ndx = (lt>=0) & (lt<m)
    plot(t[ndx], lt[ndx], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


def triangulate_point(x1, x2, P1, P2):
    '''使用最小二乘解，绘制点对的三角剖分'''
    M = zeros((6, 6))
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -x1
    M[3:, 5] = -x2

    U, S, V = linalg.svd(M)
    X = V[-1, :4]

    return X / X[3]


def triangulate(x1, x2, P1, P2):
    '''x1 和 x2 (3*n 的齐次坐标表示) 中点的二视图三角剖分'''
    n = x1.shape[1]
    if x2.shape[1] != n:
       raise ValueError("Number of points don't match")

    X = [triangulate_point(x1[:,i], x2[:,i], P1, P2) for i in range(n)]
    return array(X).T


def compute_P(x, X):
    '''由二维-三维对应对（齐次坐标表示）计算照相机矩阵'''
    n = x.shape[1]
    if X.shape[1] != n:
       raise ValueError("Number of points don't match.")

    # 创建用于计算 DLT 解的矩阵
    M = zeros((3*n, 12+n))
    for i in range(n):
        M[3*i, 0:4] = X[:, i]
        M[3*i+1, 4:8] = X[:, i]
        M[3*i+2, 8:12] = X[:, i]
        M[3*i:3*i+3, i+12] = -x[:, i]

    U, S, V = linalg.svd(M)

    return V[-1, :12].reshape((3, 4))


def skew(a):
    '''生成反对称矩阵 A，使得对于每个 v 有 a x v = Av '''
    return array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])


def compute_P_from_fundamental(F):
    '''从基础矩阵中计算第二个照相机矩阵（假设 P1 = [I 0]）'''
    e = compute_epipole(F.T) # 左极点
    Te = skew(e)
    return vstack((dot(Te.F).T, e)).T


def compute_P_from_essential(E):
    '''从本质矩阵中计算第二个照相机矩阵（假设 P1 = [I 0]）
       返回 4 个可能的照相机矩阵列表'''

    # 保证 E 的秩为 2
    U, S, V = svd(E)
    if det(dot(U, V)) < 0:
        V = -V
    E = dot(U, dot(diag([1,1,0]), V))

    # 创建矩阵(Hartley)
    Z = skew([0,0,-1])
    W = array([[0,-1,0], [1,0,0], [0,0,1]])

    # 返回所有 4 个解
    P2 = [vstack((dot(U, dot(W, V)).T, U[:,2])).T,
          vstack((dot(U, dot(W, V)).T, -U[:,2])).T,
          vstack((dot(U, dot(W.T, V)).T, U[:,2])).T,
          vstack((dot(U, dot(W.T, V)).T, -U[:,2])).T,
    ]

    return P2


def compute_fundamental_normalized(x1, x2):
    '''使用归一化的八点算法，由对应点(x1, x2 3*n 的数组)计算基础矩阵'''
    n = x1.shape[1]
    if x2.shape[1] != n:
       raise ValueError("Number of points don't match.")

    # 归一化图像坐标 TODO 归一化的方法
    x1 = x1 / x1[2]
    mean_1 = mean(x1[:2], axis=1)
    S1 = sqrt(2) / std(x1[:2])
    T1 = array([[S1, 0, -S1*mean_1[0]], [0, S1, -S1*mean_1[1]], [0, 0, 1]])
    x1 = dot(T1, x1)

    x2 = x2 / x2[2]
    mean_2 = mean(x2[:2], axis=1)
    S2 = sqrt(2) / std(x2[:2])
    T2 = array([[S2, 0, -S2*mean_2[0]], [0, S2, -S2*mean_2[1]], [0, 0, 1]])
    x2 = dot(T2, x2)

    # 使用归一化的坐标计算 F
    F = compute_fundamental(x1, x2)

    # 反归一化
    F = dot(T1.T, dot(F, T2))

    return F / F[2, 2]


def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-6):
    '''使用 RANSAC 方法从点对应中稳健地估计基础矩阵 F
       输入：齐次坐标表示的点 x1, x2 (3*n 的数组)'''

    from . import ransac
    data = vstack((x1, x2))
    # 计算 F，并返回正确点索引
    F, ransac_data = ransac.ransac(data.T, model, 8, maxiter, match_threshold, 20, return_all=True)
    return F, ransac_data['inliers']
