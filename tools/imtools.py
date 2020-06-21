'''
一些经常使用的图像操作。
'''

import os
from PIL import Image
from pylab import array, uint8

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def imresize(im, sz):
    '''使用 PIL 对象重新定义图像数组的大小'''
    from PIL import Image
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))


def hisreq(im, nbr_bins=256):
    '''对一幅图像进行直方图均衡化'''

    # 计算图像的直方图, 返回每一个 bin 中的概率密度
    imhist, bins = histogram(im.flatten(), nbr_bins, density=True)
    # cumulative distribution function
    cdf = imhist.cumsum()
    # 归一化
    cdf = 255 * cdf / cdf[-1]
    # 使用累计分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


def compute_average(imlist):
    '''计算图像列表的平均图像'''
    averageim = array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            pass
            # print(imname + '...skipped', Image.open(imname).size)
    averageim /= len(imlist)

    return array(averageim, 'uint8')
