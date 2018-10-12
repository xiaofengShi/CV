#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-10 12:08:06
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-10 12:08:06
'''
对公式进行预测，该文件对应着模型的加载和公式预测
'''
import os
import shutil
import sys
from math import *
# parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parentdir)
sys.path.append(os.path.dirname(__file__))
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import data_loaders
import tflib
import tflib.network
import tflib.ops
import tflib.optimizer
from config_formula import cfg as cfg
from dataset.data_find_all_dirs import GetFileFromThisRootDir
from tflib.ops import im2latexAttention
import scipy

slim = tf.contrib.slim

D = 512
H = 20
W = 50
IMG_PATH = cfg.IMG_DATA_PATH
PROPERTIES = cfg.PROPERTIES
# PROPERTIES = '/Users/xiaofeng/Desktop/properties_generate_enhance.npy'

# 初始参数设置


# 加载存储的符号文件并转化成本list形式
properties = np.load(cfg.PROPERTIES).tolist()


class Net():
    def __init__(self):
        self.net_name = 'resnet_v2_50'
        self.num_layers = 0
        self.learning_style = 'exponential'
        if self.net_name not in cfg.NET_LIST:
            print('net_name:{} is wrong!!!!!!')
            sys.exit()
        #===================================占位符=============================================#
        self.X = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32, name='input_img')
        self.seqs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='label')
        self.mask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='mask')
        self.input_seqs = self.seqs[:, :-1]
        self.target_seqs = self.seqs[:, 1:]
        self.net_func = tflib.network.net_fun(net_name=self.net_name, num_classes=None)
        self.ctx, _ = self.net_func(inputs=self.X)
        self.D = 512
        self.H = 20
        self.W = 50
        #==============================这里只加载网络中存在的参数====================================#
        # 如果不是第一次训练，那么就进行公开数据集的权重加载
        # =========resnet 之后连接卷积层，与LSTM连接，保证卷积层的输出通道数目============================#
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.zeros_initializer(), padding='SAME'):
            self.ctx = slim.conv2d(self.ctx, cfg.MODEL.FEATURE, [3, 3], scope='CNN_OUT')

        """ embeding shape 为(batch,label_length,embeding_dim) """
        self.emb_seqs = tflib.ops.Embedding(name='Embedding', n_symbols=cfg.VOCABLARY_SIZE,
                                            output_dim=cfg.MODEL.DIMS_INPUT, indices=self.input_seqs)
        """
        添加attention的encode——decode
        seq2seq模型
        根据预设的层数进行RNN网络的选择
        输出为batch_size,label_length,dims_out
        对于存在attention层，dims_out=dims_attention
        """
        self.rnn_out, _ = tflib.ops.im2latexAttention(
            name='AttLSTM', inputs=self.emb_seqs, ctx=self.ctx, input_dim=cfg.MODEL.DIMS_INPUT,
            ENC_DIM=cfg.MODEL.DIMS_HIDDEN, DEC_DIM=cfg.MODEL.DIMS_ATTENTION, D=self.D, H=self.H, W=self.W)
        # 进行全连接输出维度转换，batch_size,label_length,dims_out变成维度batch_size,label_length,vocab_size
        self.logits = tflib.ops.Linear('MLP.1', self.rnn_out, cfg.MODEL.DIMS_ATTENTION, cfg.VOCABLARY_SIZE)
        """ 使用输入的字符预测当前最后一个字符 """
        self.output_index = tf.argmax(tf.nn.softmax(self.logits[:, -1]), axis=1)
        """ 使用输入的字符进行错位预测 """
        self.output_index_dislocation = tf.argmax(tf.reshape(self.logits, [-1, cfg.VOCABLARY_SIZE]), axis=1)


""" 进行config的设置 """
config = tf.ConfigProto(device_count={"CPU": cfg.CPU_NUMS}, intra_op_parallelism_threads=cfg.CPU_THREADS)
if cfg.GPU:
    config.gpu_options.per_process_gpu_memory_fraction = cfg.GPU_PERCENTAGE


def load_formula_model(ckpt_path, sess3, graphic3):
    with sess3.as_default():
        with graphic3.as_default():
            net = Net()
            saver_formula = tf.train.Saver(tf.global_variables())
            print('Restore the weight files from: {}'.format(ckpt_path))
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                print("Tensor_name is : ", key)
            saver_formula.restore(sess3, ckpt.model_checkpoint_path)
            print("Load formula recognise session done")
    return sess3, saver_formula, net


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation,
                                  np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation,
                                  np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])):min(ydim - 1, int(pt3[1])),
                         max(1, int(pt1[0])):min(xdim - 1, int(pt3[0]))]
    # height,width=imgOut.shape[:2]
    return imgOut


DEBUG = True


def predict_img_latex(img, sess, net, text_recs, adjust=False):
    # imgs = np.asarray(img.convert('RGB'))
    imgs = img
    count = 1
    index = 0
    results = {}
    xDim, yDim = imgs.shape[1], imgs.shape[0]
    for index, rec in enumerate(text_recs):
        results[index] = [rec, ]
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

        partImg = dumpRotateImage(imgs, degree, pt1, pt2, pt3, pt4)

        if DEBUG:
            import copy
            print('partImg_test_shape:', np.shape(partImg))
            cv2.imwrite(
                '/Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/test_img/' + str(count) + '.jpg',
                partImg)
            count += 1
        width = np.shape(partImg)[1]
        partImg = np.asarray([np.asarray(partImg)], dtype=np.float32)
        # The predictde length was baed on the img's width
        char_length = int(width / 2)
        print('part_img', np.shape(partImg))
        # char_length = 300
        # Convert NCHW to NHWC
        inp_seqs = np.zeros((cfg.TEST.BATCH_SIZE, char_length)).astype('int32')
        inp_seqs[0, :] = properties['char_to_idx']['#START']
        tflib.ops.ctx_vector = []

        def idx_to_chars(Y): return ' '.join(map(lambda x: properties['idx_to_char'][x], Y))
        print()
        for i in range(1, char_length):
            inp_seqs[:, i] = sess.run(
                [net.output_index], feed_dict={net.X: partImg, net.input_seqs: inp_seqs[:, :i]})
        str_ori = idx_to_chars(inp_seqs.flatten().tolist())
        print('str_ori', str_ori)
        formula = idx_to_chars(inp_seqs.flatten().tolist()).split('#END')[0].split('START')[-1]
        results[index].append(formula)
    return results
