# -*- coding: utf-8 -*-
""" 预处理模块，提供三个实用函数
* get_X 接受一个图片文件路径，返回预处理后的图片向量
* get_Y 接受一个图片文件路径，返回验证码文本
* fetch_data 接受一个数字和文件目录，返回相应数量的数据集
"""
import os
import glob

from config import IMAGE_HEIGHT, IMAGE_WIDTH, CHARSET
from PIL import Image
import numpy as np


def pixelFilter(img, threshold, replace, mode):
    """Replace some pixels and return a new one

    Used to remove some noise

    :param img: an numpy.array
    :param threshold: usually a tuple pixels compare to.
    :param replace: pixels used to replace
    :param mode: 1 means if pixels larger than threshold then do, 0 means less.

    Usage:
      >>> img_ = np.array(img)
      >>> img_ = pixelFilter(np.array(img2), (15, 15, 15), (255, 255, 255), 1)
      >>> img_ = Image.fromarray(img_)
    """
    if mode == 1:
        img_ = np.where(img>threshold, replace, img)
    elif mode == 0:
        img_ = np.where(img<threshold, replace, img)
    else:
        raise Exception("mode is either 1 or 0")
    return np.array(img_, dtype='uint8', copy=False)

def imageTransform(img, size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    # resize
    img = img.resize(size)

    # gray
    img = img.convert('L')

    # filter
    img_ = np.array(img)
    img_ = pixelFilter(img_, 60, 255, 1)
    img = Image.fromarray(img_)

    # normalize
    img = np.array(img).flatten() / 255

    return img

def text2int(text, charset=CHARSET):
    return charset.index(text)


def int2text(index, charset=CHARSET):
    return charset[index]


def get_X(path):
    img = Image.open(path)
    img = imageTransform(img)
    return img


def get_Y(path):
    basename = os.path.basename(path)
    text = basename[0]
    int_ = text2int(text)
    return int_


def fetch_data(num, imgdir):
    """randomly select `num` image"""
    pathnames = glob.glob(imgdir+"*")
    index = np.arange(len(pathnames))
    np.random.shuffle(index)
    pathnames = [pathnames[i] for i in index[:num]]
    X = [get_X(path) for path in pathnames]
    y = [get_Y(path) for path in pathnames]
    return X, y