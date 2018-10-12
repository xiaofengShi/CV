#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-12 12:03:05
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-12 12:03:05
'''
将数据集创建成npy的格式
'''
import datetime
import os
import random
import re
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)


import numpy as np
from IPython.display import Image, Latex, Math, display
from PIL import Image
from tqdm import tqdm

from config_formula import cfg as cfg

vocab = open(cfg.VOCAB_PATH).readlines()
formula = open(cfg.FORMULA_PATH, 'r').readlines()
char_to_idx = {x.split('\n')[0]: i for i, x in enumerate(vocab)}
char_to_idx['#UNK'] = len(char_to_idx)
char_to_idx['#START'] = len(char_to_idx)
char_to_idx['#END'] = len(char_to_idx)


idx_to_char = {y: x for x, y in char_to_idx.items()}
# create properities
properties = {}
properties['vocab_size'] = len(vocab)
properties['vocab'] = vocab
properties['char_to_idx'] = char_to_idx
properties['idx_to_char'] = idx_to_char
# 保存类别为npy
print('saving properties!!')
np.save(os.path.join(cfg.PREPARED, 'properties'), properties)

# 保存训练npy
print(len(char_to_idx))
print(char_to_idx)
for set in cfg.DATASET_LIST:
    print('current set is:', set)
    file_list = open(os.path.join(cfg.DATA_LABEL_PATH, set + "_filtered.lst"), 'r').readlines()
    set_list, missing = [], {}
    for i, line in enumerate(file_list):
        # file_list的形式为   7944775fc9.png 32771
        # form得到formulae对应的行数的位置
        form = formula[int(line.split()[1])].strip().split()
        # out_form最开始的位置为['#START']-504,不存在于vocb中的字符为'#UNK'，结尾使用'#END'
        out_form = [char_to_idx['#START']]
        # out_form = []
        for char in form:
            try:
                out_form += [char_to_idx[char]]
            except:
                if char not in missing.keys():
                    print(char, " not found!")
                    missing[char] = 1
                else:
                    missing[char] += 1
                out_form += [char_to_idx['#UNK']]

        out_form += [char_to_idx['#END']]
        # set_list中存储为[图像名称，对应的图像的label]
        set_list.append([line.split()[0], out_form])
    print('set_list', set_list[-1])
    buckets = {}
    file_not_found_count = 0
    file_not_found = []
    for img, label in tqdm(set_list):
        if os.path.exists(os.path.join(cfg.IMG_DATA_PATH, img)):
            img_shp = Image.open(os.path.join(cfg.IMG_DATA_PATH, img)).size
            try:
                buckets[img_shp] += [(img, label)]
            except:
                buckets[img_shp] = [(img, label)]
        else:
            file_not_found_count += 1
            file_not_found.append(img)
    Info_out = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '   ' + 'Num files found in %s set: %d/%d' % (
        set, len(set_list) - file_not_found_count, len(set_list)) + '\n' + 'Missing char:' + str(
        missing) + '\n' + 'Missing files:' + str(file_not_found) + '\n' + 'size_list:'+str(buckets.keys()) + '\n'
    with open(os.path.join(cfg.PREPARED, 'GenerateNPY.txt'), 'a') as txt:
        txt.writelines(Info_out)

    txt.close()
    # 保存成npy格式文件
    print('write files to:', os.path.join(cfg.PREPARED, set + '_buckets'))
    np.save(os.path.join(cfg.PREPARED, set + '_buckets'), buckets)
