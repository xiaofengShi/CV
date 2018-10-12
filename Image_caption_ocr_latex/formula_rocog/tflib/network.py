#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-26 19:16:19
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-26 19:16:19

import os
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])
import tflib
import tflib.ops
import tensorflow as tf
import numpy as np
from resnet_utils import resnet_arg_scope
import vgg
import resnet_v2

import functools

slim = tf.contrib.slim


def alex_net(inp, DIM=512):
    X = tf.nn.relu(
        tflib.ops.conv2d(
            'conv1', inp, 11, 4, 1, 96, bias=True, batchnorm=False,
            pad='SAME'))
    X = tflib.ops.max_pool('pool1', X, k=3, s=2)
    X = tflib.ops.norm('norm1', X, lsize=5)

    X = tf.nn.relu(
        tflib.ops.conv2d(
            'conv2', X, 5, 1, 96, 256, bias=True, batchnorm=False, pad='SAME'))
    X = tflib.ops.max_pool('pool2', X, k=3, s=2)
    X = tflib.ops.norm('norm2', X, lsize=5)

    X = tf.nn.relu(
        tflib.ops.conv2d(
            'conv3', X, 3, 1, 256, 384, bias=True, batchnorm=False,
            pad='SAME'))
    X = tf.nn.relu(
        tflib.ops.conv2d(
            'conv4', X, 3, 1, 384, 384, bias=True, batchnorm=False,
            pad='SAME'))
    X = tf.nn.relu(
        tflib.ops.conv2d(
            'conv5', X, 3, 1, 384, 256, bias=True, batchnorm=False,
            pad='SAME'))
    X = tflib.ops.max_pool('pool5', X, k=3, s=2)
    X = tflib.ops.norm('norm5', X, lsize=5)

    X = tf.nn.relu(
        tflib.ops.Linear('fc6', tf.reshape(X, [tf.shape(X)[0], -1]), 32768,
                         4096))
    X = tf.nn.dropout(X, 0.5)

    X = tf.nn.relu(tflib.ops.Linear('fc7', X, 4096, 4096))
    X = tf.nn.dropout(X, 0.5)

    X = tflib.ops.Linear('fc8', X, 4096, DIM)

    return X


def alex_net_att(inp):
    X = tf.nn.relu(
        tflib.ops.conv2d('conv1', inp, 11, 4, 1, 96, bias=True, batchnorm=False, pad='SAME'))
    X = tflib.ops.max_pool('pool1', X, k=3, s=2)
    X = tflib.ops.norm('norm1', X, lsize=5)

    X = tf.nn.relu(
        tflib.ops.conv2d('conv2', X, 5, 1, 96, 256, bias=True, batchnorm=False, pad='SAME'))
    X = tflib.ops.max_pool('pool2', X, k=3, s=2)
    X = tflib.ops.norm('norm2', X, lsize=5)

    X = tf.nn.relu(
        tflib.ops.conv2d('conv3', X, 3, 1, 256, 384, bias=True, batchnorm=False, pad='SAME'))
    X = tf.nn.relu(
        tflib.ops.conv2d('conv4', X, 3, 1, 384, 384, bias=True, batchnorm=False, pad='SAME'))
    X = tf.nn.relu(
        tflib.ops.conv2d('conv5', X, 3, 1, 384, 256, bias=True, batchnorm=False, pad='SAME'))

    return X


def vgg16(X, num_feats=64):
    X = tf.nn.relu(
        tflib.ops.conv2d('conv1_1', X, 3, 1, 1, num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tf.nn.relu(
        tflib.ops.conv2d('conv1_2', X, 3, 1, num_feats, num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tflib.ops.max_pool('pool1', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv2_1', X, 3, 1, num_feats,
                                    2 * num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv2_2', X, 3, 1, 2 * num_feats,
                                    2 * num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tflib.ops.max_pool('pool2', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv3_1', X, 3, 1, 2 * num_feats,
                                    4 * num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv3_2', X, 3, 1, 4 * num_feats,
                                    4 * num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv3_3', X, 3, 1, 4 * num_feats,
                                    4 * num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tflib.ops.max_pool('pool3', X, k=2, s=2)

    X = tf.nn.relu(
        tflib.ops.conv2d('conv4_1', X, 3, 1, 4 * num_feats, 8 * num_feats, bias=True, batchnorm=False,
                         pad='SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv4_2', X, 3, 1, 8 * num_feats,
                                    8 * num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv4_3', X, 3, 1, 8 * num_feats,
                                    8 * num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tflib.ops.max_pool('pool4', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv5_1', X, 3, 1, 8 * num_feats,
                                    8 * num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv5_2', X, 3, 1, 8 * num_feats,
                                    8 * num_feats, bias=True, batchnorm=False, pad='SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv5_3', X, 3, 1, 8 * num_feats,
                                    8 * num_feats, bias=True, batchnorm=False, pad='SAME'))

    return X


def im2latex_cnn(X, num_feats, bn, train_mode=True):
    # 数据归一化，将数据调整到[-1,1]之间
    X = X - 128.
    X = X / 128.

    X = tf.nn.relu(tflib.ops.conv2d('conv1', X, 3, 1, 1, num_filters=num_feats, pad='SAME', bias=False))
    X = tflib.ops.max_pool('pool1', X, k=2, s=2)
    X = tf.nn.relu(tflib.ops.conv2d('conv2', X, 3, 1, num_feats, num_feats * 2, pad='SAME', bias=False))
    X = tflib.ops.max_pool('pool2', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv3', X, 3, 1, num_feats * 2, num_feats * 4,
                                    batchnorm=bn, is_training=train_mode, pad='SAME', bias=False))

    X = tf.nn.relu(tflib.ops.conv2d('conv4', X, 3, 1, num_feats * 4, num_feats * 4, pad='SAME', bias=False))
    X = tflib.ops.max_pool('pool4', X, k=(1, 2), s=(1, 2))

    X = tf.nn.relu(tflib.ops.conv2d('conv5', X, 3, 1, num_feats * 4, num_feats * 8,
                                    batchnorm=bn, is_training=train_mode, pad='SAME', bias=False))
    X = tflib.ops.max_pool('pool5', X, k=(2, 1), s=(2, 1))

    X = tf.nn.relu(tflib.ops.conv2d('conv6', X, 3, 1, num_feats * 8, num_feats * 8,
                                    batchnorm=bn, is_training=train_mode, pad='SAME', bias=False))

    return X


# 相对高端一些的实现方式 ^_^
# 首先根据网络的名称获取网络方程
# 之后
net_map = {
    'vgg_16': vgg.vgg_16,
    'vgg_19': vgg.vgg_19,
    'resnet_v2_50': resnet_v2.resnet_v2_50,
    'resnet_v2_101': resnet_v2.resnet_v2_101,
    'resnet_v2_152': resnet_v2.resnet_v2_152,
    'resnet_v2_200': resnet_v2.resnet_v2_200
}
arg_map = {
    'vgg_16': vgg.vgg_arg_scope,
    'vgg_19': vgg.vgg_arg_scope,
    'resnet_v2_50': resnet_arg_scope,
    'resnet_v2_101': resnet_arg_scope,
    'resnet_v2_152': resnet_arg_scope,
    'resnet_v2_200': resnet_arg_scope
}


def net_fun(net_name,
            num_classes=None,
            weight_decay=0.00004,
            is_training=True,
            global_pool=False):
    if net_name not in net_map:
        raise ValueError('Name of network unknown %s' % net_name)

    func = net_map[net_name]

    @functools.wraps(func)
    def network_fn(inputs, **kwargs):
        arg_scope = arg_map[net_name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(
                inputs,
                num_classes,
                is_training=is_training,
                global_pool=global_pool,
                **kwargs)

    return network_fn


# 比较low的实现方式
def net_fatory(net_name, inputs, train_model, FC=False):
    if net_name == 'vgg_16':
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net, end_points = vgg.vgg_16(
                inputs, num_classes=None, is_training=train_model, fc_flage=FC)
    elif net_name == 'vgg_19':
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net, end_points = vgg.vgg_19(
                inputs, num_classes=None, is_training=train_model, fc_flage=FC)
    elif net_name == 'resnet_v2_50':
        with slim.arg_scope(resnet_arg_scope()):
            net, end_points = resnet_v2.resnet_v2_50(
                inputs=inputs,
                num_classes=None,
                is_training=train_model,
                global_pool=False)
    elif net_name == 'resnet_v2_152':
        with slim.arg_scope(resnet_arg_scope()):
            net, end_points = resnet_v2.resnet_v2_152(
                inputs=inputs,
                num_classes=None,
                is_training=train_model,
                global_pool=False)

    return net, end_points

# def Seq2Seq_attention():
