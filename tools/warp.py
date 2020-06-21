from PIL import Image
from numpy import *
from pylab import *
from scipy import ndimage
from matplotlib.tri import Triangulation

from . import homography
from importlib import reload
reload(homography)
from . import homography

def image_in_image(im1, im2, tp):
    '''使用仿射变换将 im1 放置在 im2 上，使 im1 图像的角和 tp 尽可能的靠近。
       tp 是齐次表示的，并且按照从左上角逆时针计算'''

    # 扭曲的点
    m, n = im1.shape[:2]
    fp = array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    # 计算仿射变换，并且将其应用于图像 im1
    H = homography.Haffine_from_points(fp, tp) # TODO
    im1_t = ndimage.affine_transform(im1, H[:2, :2], (H[0, 2], H[1, 2]), im2.shape[:2])
    alpha = (im1_t > 0)

    return (1-alpha)*im2 + alpha*im1_t


def alpha_for_triangle(points, m, n):
    '''对于带有由 points 定义角点的三角形，创建大小为 (m,n) 的 alpha 图
       (在归一化的齐次坐标意义下)'''

    alpha = zeros((m, n))
    for i in range(min(points[0]), max(points[0])):
        for j in range(min(points[1]), max(points[1])):
            x = linalg.solve(points, [i, j, 1])
            if min(x) > 0:   # 所有系数都大于零
                alpha[i, j] = 1
    return alpha


def triangulate_points(x, y):
    '''二维点的 Delaunay 三角划分'''
    return Triangulation(x, y).triangles


def panorama(H, fromim, toim, padding=2400, delta=2400):
    '''使用单应性矩阵 H，协调两幅图像，创建水平全景图像。结果为一幅和 toim
       具有相同高度的图像。padding 指定填充像素的数目，delta 指定额外的平移量。'''

    # 检查图像是灰度图像，还是彩色图像
    is_color = len(fromim.shape) == 3

    # 用于 geometric_transform() 的单应性变换
    def transf(p):
        '''将像素和 H 相乘，然后对齐次坐标归一化来实现像素间的映射'''
        p2 = dot(H, [p[0], p[1], 1])
        return (p2[0]/p2[2], p2[1]/p2[2])

    # 通过查看 H 中的平移量，决定应该将图像填补到左边还是右边
    if H[1,2] < 0: # fromim 在左边
        print('warp - right')
        # 变换 fromim
        if is_color:
            # 在目标图像的右边填充 0
            toim_t = hstack((toim, zeros((toim.shape[0], padding, 3))))
            fromim_t = zeros((toim.shape[0], toim.shape[1]+padding, toim.shape[2]))
            for col in range(3):
                fromim_t[:, :, col] = ndimage.geometric_transform(
                                        fromim[:, :, col],
                                        transf,
                                        (toim.shape[0], toim.shape[1]+padding))
        else:
            # 在目标图像的右边填充 0
            toim_t = hstack((toim, zeros((toim.shape[0], padding))))
            fromim_t = ndimage.geometric_transform(fromim, transf, (toim.shape[0], toim.shape[1]+padding))
    else:
        print('warp - left')
        # 为了补偿填充效果，在左边加入平移量
        H_delta = array([[1,0,0], [0,1,-delta], [0,0,1]])
        H = dot(H, H_delta)
        # fromim 变换
        if is_color:
            # 在目标图片左边填充 0
            toim_t = hstack((zeros((toim.shape[0], padding, 3))), toim)
            fromim_t = zeros((toim.shape[0], toim.shape[1]+padding, toim.shape[2]))
            for col in range(3):
                fromim_t[:, :, col] = ndimage.geometric_transform(
                                        fromim[:, :, col],
                                        transf,
                                        (toim.shape[0], toim.shape[1]+padding))
        else:
            # 在目标图片左边填充 0
            toim_t = hstack((zeros((toim.shape[0], padding))), toim)
            fromim_t = ndimage.geometric_transform(fromim, transf, (toim.shape[0], toim.shape[1]+padding))

    # 协调后返回 (将 fromim 放置在 toim 上)
    if is_color:
        # 所有非黑色像素
        alpha = ((fromim_t[:,:,0] * fromim_t[:,:,1] * fromim_t[:,:,2]) > 0)
        for col in range(3):
            toim_t[:,:,col] = fromim_t[:,:,col]*alpha + toim_t[:,:,col]*(1-alpha)
    else:
        alpha = (fromim_t > 0)
        toim_t = fromim_t * alpha + toim_t * (1-alpha)
    return toim_t

