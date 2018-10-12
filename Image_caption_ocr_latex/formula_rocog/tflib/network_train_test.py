'''
File: Net_train_test.py
Project: formula_rocog
File Created: Monday, 2nd July 2018 3:40:09 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Tuesday, 3rd July 2018 11:58:46 am
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf

import tflib
import tflib.ops
import tflib.network

from config_formula import cfg as cfg


class NET_TRAIN(object):
    def __init__(self, trainable=True):
        self.img = tf.placeholder(shape=(None, None, None, None), dtype=tf.float32, name='input_img')
        self.seqs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='label')
        self.mask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='mask')
        self.input_seqs = self.seqs[:, :-1]
        self.target_seqs = self.seqs[:, 1:]
        self.trainable = trainable
        self.D = 512
        self.H = 20
        self.W = 50
        self.model()

    def model(self):
        self.cnn_out = tflib.network.im2latex_cnn(self.img)
        self.emb_seqs = tflib.ops.Embedding(name='Embedding', n_symbols=cfg.VOCABLARY_SIZE,
                                            output_dim=cfg.MODEL.DIMS_INPUT, indices=self.input_seqs)
        self.rnn_out, _ = tflib.ops.im2latexAttention(
            name='AttLSTM', inputs=self.emb_seqs, ctx=self.cnn_out, input_dim=cfg.MODEL.DIMS_INPUT,
            ENC_DIM=cfg.MODEL.DIMS_HIDDEN, DEC_DIM=cfg.MODEL.DIMS_ATTENTION, D=self.D, H=self.H, W=self.W)
        self.logits = tflib.ops.Linear('logits', self.rnn_out, cfg.MODEL.DIMS_ATTENTION, cfg.VOCABLARY_SIZE)
        # m模型预测的结果
        self.prediction = tf.argmax(tf.nn.softmax(self.logits[:, -1]), axis=1)

        self.loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(
            self.logits, [-1, cfg.VOCABLARY_SIZE]), labels=tf.reshape(self.seqs[:, 1:], [-1])), [tf.shape(self.img)[0], -1])

        self.output = tf.reshape(self.logits, [-1, cfg.VOCABLARY_SIZE])
        # 找到每一列的最大值，返回的维度为output——index为(batch_size*label_length)
        self.output_index = tf.to_int32(tf.argmax(input=self.output, axis=1, name='argmax'))
        # 将(batch_size,label_length)reshape成(batch_size*label_length)
        self.true_labels = tf.reshape(self.seqs[:, 1:], [-1])

        self.correct_prediction = tf.equal(self.output_index, self.true_labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.mask_mult = tf.to_float(self.mask[:, 1:])

        self.loss_total = tf.reduce_sum(self.loss * self.mask_mult) / tf.reduce_sum(self.mask_mult)
