#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-01 12:12:42
# Last Modified by: xiaofeng
# Last Modified time: 2018年04月25日17:02:28

'''
搭建tensorflow版本的crnn，进行训练
'''
import os
import sys
import pprint
import time
from functools import reduce
from operator import mul
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import ctc_ops

import data_loaders_ctc
import tfutils
import tfutils.config_crnn as cfg
import tfutils.utils as utils
from train_test.train_modul import train_net


if __name__ == '__main__':
    # 将text.yml的配置与默认config中的默认配置进行合并
    # cfg_from_file(
    #     '/Users/xiaofeng/Code/Github/graphic/Chinese-OCR/ctpn/ctpn/text.yml')
    # print('Using config:~~~~~~~~~~~~~~~~')
    # pprint.pprint(cfg)
    # 根据给定的名字，得到要加载的数据集
    # imdb = get_imdb('voc_2007_trainval')
    '''
    NET_LIST = ['vgg_16', 'vgg_19', 'resnet_v2_50', 'resnet_v2_152']
    LEARNING_STYLE = ['exponential', 'fixed', 'polynomial']    
    OPTIMIZER = ['adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'rmsprop', 'sgd']
    '''
    dataset_name_list = cfg.DATASET_LIST
    print('Dataset list is :', dataset_name_list)
    print('Load the train and validate dataset.')
    train_imdb = data_loaders_ctc.data_iterator(
        'train', batch_size=cfg.BATCH_SIZE)
    validate_imdb = data_loaders_ctc.data_iterator(
        'validate', batch_size=cfg.BATCH_SIZE)
    output_dir = cfg.MODEL_SAVED
    log_dir = cfg.SUMMARY_PATH
    pretrained_dir = cfg.PRETRAINED

    cnn_net_name = 'resnet_v2_50'
    optimizer_name = 'adam'
    loss_function = 'softmax'
    learning_style = 'exponential'

    print('Output will be saved to `{:s}`'.format(output_dir))
    print('Logs will be saved to `{:s}`'.format(log_dir))
    print('Pretrained path is  `{:s}`'.format(pretrained_dir))
    print('cnn  network name is :', cnn_net_name)
    print('loss function is :', loss_function)

    # device_name = '/gpu:0'
    device_name = '/cpu:0'
    # print(device_name)
    print('get the net work basd name')

    network = utils.get_network(
        name='CRNN_train', cnn_name=cnn_net_name, loss_function=loss_function,
        trainable=True, flag_embed=False)

    train_net(
        network,
        cnn_net_name,
        train_imdb,
        validate_imdb,
        optimizer_name,
        learning_style,
        output_dir=output_dir,
        log_dir=log_dir,
        pretrained_path=pretrained_dir,
        pretrained_model=None,
        # '/Users/xiaofeng/Code/Github/dataset/CHINESE_OCR/ctpn/pretrain/VGG_imagenet.npy',
        max_iters=180000,
        restore=bool(int(0)))
