#!/usr/bin/env python
# -*- coding: utf-8 -*-
_Author_ = 'xiaofeng'

import tensorflow as tf
import config as cfg

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def tiny_net_arg_scope(weight_decay=0.0005,
                       stddev=0.1,
                       batch_norm_var_collection='moving_vars'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=trunc_normal(stddev),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                normalizer_params=batch_norm_params) as sc:
            return sc


def tiny_net(inputs,
             num_classes=cfg.CLASSES,
             is_training=True,
             dropout_keep_prob=0.5,
             spatial_squeeze=True,
             scope='vgg_16',
             fc_conv_padding='VALID',
             global_pool=False):
    with tf.variable_scope(scope, 'tiny_net', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 76, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 3, slim.conv2d, 152, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 304, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 608, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.conv2d(net, 456, [1, 1], scope='conv5')
            net = slim.conv2d(net, 456, [1, 4], padding=fc_conv_padding, scope='conv6')
            net = slim.conv2d(net, 456, [3, 1], padding=fc_conv_padding, scope='conv7')
            # Use conv2d instead of fully_connected layers.
            # net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout1')
            net = slim.conv2d(net, 456, [1, 1], scope='fc1')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            if num_classes:
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout2')
                net = slim.conv2d(net, num_classes, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc2')
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc3/squeezed')
                end_points[sc.name + '/fc3'] = net

            return net, end_points
