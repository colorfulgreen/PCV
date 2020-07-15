from scipy import linalg
from numpy import *

class Camera(object):
    '''表示针孔照相机的类'''

    def __init__(self, P):
        '''初始化 P = K[R|t] 的照相机模型'''
        self.P = P
        self.K = None
        self.R = None
        self.t = None
        self.c = None # TODO 照相机中心

    def project(self, X):
        '''X(4*n的数组)的投影点，并且进行坐标归一化'''
        # TODO 坐标归一化
        x = dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]
        return x

