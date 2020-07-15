from . import camera
from PIL import Image
from numpy import *

def load_vggdata():
    # 载入前两幅图像
    im1 = array(Image.open('data/Merton1/001.jpg'))
    im2 = array(Image.open('data/Merton1/002.jpg'))

    # 载入三个视图中的所有图像特征点(Harris 角点)
    points2D = [loadtxt('data/Merton1/00%s.corners' % str(i+1)).T for i in range(3)]

    # 载入对应不同视图图像点重建后的三维点
    points3D = loadtxt('data/Merton1/p3d').T

    # load correspondences
    # 因为并不是所有的点都可见，或都能够成功匹配到所有的视图，所以对应数据里包含了缺失的数据，填充为 -1
    corr = genfromtxt('data/Merton1/nview-corners', dtype='int', missing_values='*')

    # load cameras to a list of Camera objects
    P = [camera.Camera(loadtxt('data/Merton1/00%s.P' % str(i+1))) for i in range(3)]

    return im1, im2, points2D, points3D, corr, P
