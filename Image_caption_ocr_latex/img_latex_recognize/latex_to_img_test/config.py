#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-01 11:37:26
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-01 11:37:26

import os

A_NOTE = '路径存储信息'
DATA_ROOT_PATH = '/Users/xiaofeng/Code/Github/dataset/formula/prepared/'
MEDEL_SAVED = '/Users/xiaofeng/Code/Github/dataset/formula/model_saved_graphic/'
IMAGE_PATH = DATA_ROOT_PATH + 'images_processed/'
VOCAB_PATH = DATA_ROOT_PATH + 'latex_vocab.txt'
LABEL_PATH = DATA_ROOT_PATH + 'formulas.norm.lst'
TRAIN_PATH = DATA_ROOT_PATH + 'train_filter.lst'
TEST_PATH = DATA_ROOT_PATH + 'test_filter.lst'
VALIDATE_PATH = DATA_ROOT_PATH + 'validate_filter.lst'
MODEL_SAVED_PATH = os.path.join(MEDEL_SAVED, 'ckpt')
SUMMARY_PATH = os.path.join(MEDEL_SAVED, 'log')
BATCH_SIZE = 20
EPOCH_NUMS = 10000
LEARNING_RATE = 0.1
MIN_LEARNING_RATE = 0.001
DISPLAY_NUMS = 10
SAVED_NUMS = 100
