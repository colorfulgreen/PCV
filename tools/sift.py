import os

from PIL import Image
from numpy import *
from pylab import *

def process_image(imagename, resultname, params='--edge-thresh 10 --peak-thresh 5'):
    '''处理一幅图像，然后将结果保存在文件中'''
    if imagename[-3:] != 'pgm':
        # 创建一个 PGM (portable graymap file format) 文件
        im = Image.open(imagename).convert('L')
        im.save('/tmp/tmp.pgm')
        imagename = '/tmp/tmp.pgm'

    cmmd = 'sift {} --output {} {}'.format(imagename, resultname, params)
    # print(cmmd)
    os.system(cmmd)


def read_features_from_file(filename):
    '''将特征从文件读到 NumPy 数组中'''
    f = loadtxt(filename)
    return f[:,:4], f[:,4:]


def write_features_to_file(filename, locs, desc):
    '''将特征位置和描述子保存到文件'''
    savetxt(filename, hstack((locs, desc)))


def plot_features(im, locs, circle=False):
    '''显示带有特征的图像
    输入：im(数组图像)，locs(每个特征的行、列、尺度和朝向)'''

    def draw_circle(c, r):
        t = arange(0, 1.01, .01)*2*pi
        x = r*cos(t) + c[0]
        y = r*sin(t) + c[1]
        plot(x, y, 'b', linewidth=2)
        imshow(im)

    if circle:
        for p in locs:
            draw_circle(p[:2], p[2]) # 圆圈的半径为特征的尺度
    else:
        plot(locs[:, 0], locs[:, 1], 'ob')
    axis('off')


def match(desc1, desc2):
    '''对于第一幅图像中的每个描述子，选取其在第二幅图像中的匹配'''

    # 将描述子向量归一化到单位长度。对于单位向量，向量乘积(不使用arccos)等价于标准欧式距离度量。
    desc1 = array([d/linalg.norm(d) for d in desc1])
    desc2 = array([d/linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = zeros((desc1_size[0], 1), 'int')
    desc2t = desc2.T
    for i in range(desc1_size[0]):
        dotprods = dot(desc1[i,:], desc2t)  # 向量点乘
        dotprods = 0.9999*dotprods          # TODO ?
        # 反余弦和反排序，返回第二幅图像中特征的索引
        indx = argsort(arccos(dotprods))

        # 检查最近邻的角度是否小于 dist_ratio 乘以第二近邻的角度
        if arccos(dotprods)[indx[0]] < dist_ratio * arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores[:,0]


def match_twosided(desc1, desc2):
    '''双向对称版本的 match()'''
    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # 去除不对称的匹配
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12
