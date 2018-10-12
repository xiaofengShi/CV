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

import numpy as np
from IPython.display import Image, Latex, Math, display
from PIL import Image
from tqdm import tqdm

import config as cfg

LOCAL = True
if not LOCAL:
    INPUT_PATH = ''
    VOCAB_PATH = '/home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/char_formula.txt'
    FORMULAE_PATH = '/home/xiaofeng/code/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal.ls'
if LOCAL:
    # VOCAB_PATH = '/Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/char_formula.txt'
    VOCAB_PATH = '/Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/char_formula_full_db.txt'
    FORMULAE_PATH = '/Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/new_char_formula_normal.ls'
    INPUT_PATH = '/Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/dataset/char_formula/'

# SET_LIST = cfg.DATASET_LIST

SPACE_INDEX = 0
SPACE_CHAR = ''

vocab = open(VOCAB_PATH).readlines()
formulae = open(FORMULAE_PATH, 'r').readlines()

char_to_idx = {x.split('\n')[0]: i + 1 for i, x in enumerate(vocab)}
char_to_idx[SPACE_CHAR] = SPACE_INDEX

idx_to_char = {y: x for x, y in char_to_idx.items()}


properties = {}
properties['vocab_size'] = len(vocab)
properties['vocab'] = vocab
properties['char_to_idx'] = char_to_idx
properties['idx_to_char'] = idx_to_char
# 保存类别为npy
print('saving properties!!')
np.save(cfg.DATA_ROOT + 'properties_ctc', properties)
# 保存训练npy
print(len(char_to_idx))
print('char_to_idx', char_to_idx)
print('idx_to_char', idx_to_char)
for set in cfg.DATASET_LIST:
    print('current set is:', set)
    file_list = open(INPUT_PATH+'dataset_char_formula_' + set + "_filter.ls", 'r').readlines()
    set_list, missing = [], {}
    for i, line in enumerate(file_list):
        # file_list的形式为   7944775fc9.png 32771
        # form得到formulae对应的行数的位置
        form = formulae[int(line.split()[1])].strip().split()
        # out_form最开始的位置为['#START']-504,不存在于vocb中的字符为'#UNK'，结尾使用'#END'
        # out_form = [char_to_idx[SPACE_CHAR]]
        out_form = []
        for char in form:
            try:
                out_form += [char_to_idx[char]]
            except:
                if char not in missing.keys():
                    print(char, " not found!")
                    missing[char] = 1
                else:
                    missing[char] += 1
                # out_form += [char_to_idx['#UNK']]
                # out_form += [char_to_idx[SPACE_CHAR]]
        # out_form += [char_to_idx[SPACE_CHAR]]
        # out_form += [char_to_idx[SPACE_CHAR]]
        # set_list中存储为[图像名称，对应的图像的label]
        set_list.append([line.split()[0], out_form])
    print('set_list', set_list[-1])
    buckets = {}
    file_not_found_count = 0
    file_not_found = []
    for img, label in tqdm(set_list):
        if os.path.exists(cfg.IMG_DATA_PATH + img):
            img_shp = Image.open(cfg.IMG_DATA_PATH + img).size
            try:
                buckets[img_shp] += [(img, label)]
            except:
                buckets[img_shp] = [(img, label)]
        else:
            file_not_found_count += 1
            file_not_found.append(img)
    Info_out = datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + '   ' + 'Num files found in %s set: %d/%d' % (
            set, len(set_list) - file_not_found_count,
            len(set_list)) + '\n' + 'Missing char:' + str(
                missing) + '\n' + 'Missing files:' + str(
                    file_not_found) + '\n' + 'size_list:', str(
                        buckets.keys()) + '\n'
    with open(cfg.DATA_ROOT + 'GenerateNPY.txt', 'a') as txt:
        txt.writelines(Info_out)

    txt.close()
    print(Info_out)
    # 保存成npy格式文件
    np.save(cfg.DATA_ROOT + set + '_buckets_ctc', buckets)
