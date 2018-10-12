#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-27 21:39:33
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-27 21:39:33

from tensorflow.python import pywrap_tensorflow
import os, sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])
print(sys.path)
os.chdir(os.getcwd())

checkpoint_path_new = '/Users/xiaofeng/Code/Github/dataset/formula/generate/model_saved/ckpt/resnet_v2_50/exponential/resnet_v2_50_exponential_trained.ckpt'
checkpoint_path_old = '/Users/xiaofeng/Code/Github/dataset/formula/generate/model_saved/ckpt/resnet_v2_50/resnet_v2_50_exponential_trained.ckpt'


def calcutate_nums(checkpoint_path):
    # checkpoint_path = '/Users/xiaofeng/Code/Github/dataset/formula/generate/model_saved/ckpt/resnet_v2_50/exponential/resnet_v2_50_exponential_trained.ckpt'
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    key_list = []
    for key in var_to_shape_map:
        # print("tensor_name: ", key)
        key_list.append(key)
        # print(reader.get_tensor(key))
    # print(len(var_to_shape_map))
    return key_list


key_new = calcutate_nums(checkpoint_path_new)
key_old = calcutate_nums(checkpoint_path_old)
# with open('/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/new.txt',
#           'w') as fff:
#     for i in key_new:
#         fff.write(i + '\n')
print('new', len(key_new))
print('old', len(key_old))
# not trained values
key_dif = [i for i in key_new if i not in key_old]
for i in key_dif:
    print('tensor name:', i)
print(len(key_dif))
# with open('/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/di.txt',
#           'w') as fff:
#     for i in key_dif:
#         fff.write(i + '\n')
# # print(i)
# # new trained
# key_difff = [i for i in key_new if i not in key_ori]
# # print(key_dif)
# print('diff', len(key_dif))
# print('\n')
# # print(key_difff)
# with open('/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/diffff.txt',
#           'w') as f:
#     for i in key_difff:
#         f.write(i + '\n')
#     # print(i)
# print('\n')
# print('difffff', len(key_difff))


