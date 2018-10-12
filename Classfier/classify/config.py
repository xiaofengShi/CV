#!/usr/bin/env python
# -*- coding: utf-8 -*-
_Author_ = 'xiaofeng'
import os
import re
from dataset.data_find_all_dirs import GetFileFromThisRootDir
import math

# data path
DATASET = '/Users/xiaofeng/Work_Guanghe/datasets/'

TRAINING_OUT = '/Users/xiaofeng/Code/Github/dataset/saveout/'

TRAIN = 'train'

TEST = 'test'

OUTPUT_TFRECORD = 'TFrecord_TXT'

MODEL_SAVED = 'MODEL_SAVED'

TRAINING_PROCESS_CONFIG = 'CONFIG_PROCESS'

LOG = 'log_files'

TRAIN_DATASET = os.path.join(DATASET, TRAIN)

TEST_DATASET = os.path.join(DATASET, TEST)

TFRECORD_SAVED_DIR = os.path.join(TRAINING_OUT, OUTPUT_TFRECORD)
if not os.path.exists(TFRECORD_SAVED_DIR):
    os.makedirs(TFRECORD_SAVED_DIR)

MODEL_OUTPUT_DIR = os.path.join(TRAINING_OUT, MODEL_SAVED)
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)

SUMMARY_SAVED = os.path.join(TRAINING_OUT, LOG)
if not os.path.exists(SUMMARY_SAVED):
    os.makedirs(SUMMARY_SAVED)

TRAINING_PROCESS_CONFIG_DIR = os.path.join(
    TRAINING_OUT, TRAINING_PROCESS_CONFIG)
if not os.path.exists(TRAINING_PROCESS_CONFIG_DIR):
    os.makedirs(TRAINING_PROCESS_CONFIG_DIR)

# TFrecord making params
Project_NAME = 'cv_graphic'

RANDOM_SEED = 4242

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL = 50, 70, 3

EXTENSION = ['png', 'jpg', 'jpeg', 'tif', 'txt']

txt_list = GetFileFromThisRootDir([TFRECORD_SAVED_DIR], EXTENSION)
classes = []
if txt_list:
    file = open(txt_list[0])
    while 1:
        lines = file.readlines(1000000)
        if not lines:
            break
        for line in lines:
            # divide = line.split(': ')
            divide = [x for x in re.split(r'[:\\\s]\s*', line) if x]
            classes.append(divide[1])
            # classes[int(divide[0])] = divide[1]
    CLASSES_NAMES = classes

if not txt_list:
    CLASSES_NAMES = None

PROPORTION = 0.2

CLASSES = 44

# train and learning rate
LEARNING_RATE = 0.01

DECAY_STEPS = 5000
# DECAY_STEPS = 5

DECAY_RATE = 0.7

STAIRCASE = True

BATCH_SIZE = 16

BATCH_SIZE_TEST = 32

SUMMARY_ITER = 5

DISPLAY_STEP = 5

SAVE_ITER = 50

EPOCH_NUMS = 2000

DATASET_NUMS = len(GetFileFromThisRootDir([TRAIN_DATASET], EXTENSION))

DATASET_NUMS_TEST = len(GetFileFromThisRootDir([TEST_DATASET], EXTENSION))

ITER = int(math.ceil(DATASET_NUMS / BATCH_SIZE))

ITER_TEST = int(math.ceil(DATASET_NUMS_TEST / BATCH_SIZE_TEST))

TEST_SHOW = 20

#############
# if there is a weight to load
match = 'checkpoint'
filelist = os.listdir(MODEL_OUTPUT_DIR)
if match in filelist:
    for i in filelist:
        if os.path.splitext(i)[-1] == '.meta':
            META_DIR = os.path.join(MODEL_OUTPUT_DIR, i)
            WEIGHT_DIR = MODEL_OUTPUT_DIR
else:
    META_DIR = None
    WEIGHT_DIR = None

'''
# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
# 学习率的衰减，z指数衰减，多项式衰减，固定衰减
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

FLAGS = tf.app.flags.FLAGS
'''
