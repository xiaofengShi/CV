# -*- coding: utf-8 -*-

import codecs

WORD1 = codecs.open("wordset/word1.txt", 'r', 'utf-8').read()
#WORD2 = codecs.open("wordset/word2.txt").read()
PUNCTUATION = codecs.open("wordset/punctuation.txt").read()

# 汉字字符集
CHARSET = WORD1 + PUNCTUATION
CHARSET_LEN = len(CHARSET)


# 图片大小，设置得小点可以提高训练速度
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64


# 图片训练集存放位置
DIR_IMAGE_TRAIN = "./data/train/"

# 图片测试集存在位置
DIR_IMAGE_TEST = "./data/test/"

# 训练好的模型存放位置
DIR_MODEL = "./models/"