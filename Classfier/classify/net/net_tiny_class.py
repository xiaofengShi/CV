#!/usr/bin/env python
# -*- coding: utf-8 -*-
_Author_ = 'xiaofeng'

import numpy as np
import tensorflow as tf
import config as cfg
from net import custome_layer

slim = tf.contrib.slim


class BUILDNET(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.image_size_width = cfg.IMAGE_WIDTH
        self.image_size_height = cfg.IMAGE_HEIGHT
        self.image_channel = cfg.IMAGE_CHANNEL
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

        self.images = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size_height, self.image_size_width,
                                      self.image_channel],
                                     )
        with slim.arg_scope(self.net_arg_scope()):
            self.logits, self.endpoint = self.net(inputs=self.images, num_classes=self.classes,
                                                  is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(tf.int32, [self.batch_size, self.classes])
            loss, accurate = self.loss_layer(self.logits, self.labels)
            self.total_loss = slim.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)
            self.accurate = accurate
            tf.summary.scalar('accurate', self.accurate)

    def net(self, inputs,
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

    def net_arg_scope(self, weight_decay=0.0005,
                      stddev=0.1,
                      batch_norm_var_collection='moving_vars'):
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
                    weights_initializer=self.trunc_normal(stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params) as sc:
                return sc

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicts, labels=labels),
                                        name='loss')
            tf.losses.add_loss(class_loss)
            tf.summary.scalar('class_loss', class_loss)
            accurate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1)), tf.float32))
            tf.summary.scalar('accurate', accurate)
        return class_loss, accurate
# def leaky_relu(alpha):
#     def op(inputs):
#         return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
#
#     return op
