# !/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-05-11 12:11:06
# Last Modified by: xiaofeng
# Last Modified time: 2018-05-11 12:11:06

'''
对ctc损失函数进行数据进行feed
'''
import glob
import re
import threading
import time

import numpy as np
import tensorflow as tf
from PIL import Image


import config as cfg

PROPERTIES = cfg.PROPERTIES

properties = np.load(PROPERTIES).tolist()

SPACE_LIST = [0, 504, 505]


def sparse_tuple_from(sequences, dtype=np.int32):

    indices = []
    values = []
    sequences = np.asarray(sequences)

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    # sprase_touple = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

    # return sprase_touple

    return indices, values, shape


def sparse_tuple_to_texts_ch(tuple):
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]
    # idx_to_chars = lambda Y: ' '.join(map(lambda x: properties['idx_to_char'][x],Y))
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        # c = ' ' if c == SPACE_INDEX else words[c]  # chr(c + FIRST_INDEX)
        # c = properties['idx_to_char'][c]
        c = '' if c in SPACE_LIST else properties['idx_to_char'][c]
        results[index] = results[index] + c

    # List of strings
    return results


def ndarray_to_text_ch(value):
    results = ''
    for i in range(len(value)):
        results += properties['idx_to_char'][value[i]]  # chr(value[i] + FIRST_INDEX)
    return results.replace('`', ' ')


def batches_per_epoch(set='train', batch_size=32):
    train_dict = np.load(cfg.DATA_ROOT + set + '_buckets.npy').tolist()
    nums = np.sum([len(train_dict[x]) for x in train_dict.keys()])
    want = int(nums / batch_size)

    return want


def data_iterator(set='train', batch_size=32):
    '''
    Python data generator to facilitate mini-batch training
    Arguments:
        set - 'train','validate','test' sets
        batch_size - integer (Usually 32,64,128, etc.)
    '''
    train_dict = np.load(cfg.DATA_ROOT + set + '_buckets_ctc.npy').tolist()
    print("Nums  of %s train data: " % set,
          np.sum([len(train_dict[x]) for x in train_dict.keys()]))

    for keys in train_dict.keys():
        train_list = train_dict[keys]
        N_FILES = (len(train_list) // batch_size) * batch_size
        print('N_FILES', N_FILES, keys)
        for batch_idx in range(0, N_FILES, batch_size):
            train_sublist = train_list[batch_idx:batch_idx + batch_size]
            imgs = []
            batch_forms = []
            for x, y in train_sublist:
                imgs.append(np.asarray(Image.open(cfg.IMG_DATA_PATH + x).convert('RGB')))
                batch_forms.append(y)
            print('batch_forms', batch_forms)
            lens = [len(obj) for obj in batch_forms]
            sprase_labels = sparse_tuple_from(batch_forms)

            # 生成label，对label进行归一化，生成统一的长度，这里面涉及要加入空白区域
            mask = np.zeros((batch_size, max(lens)), dtype=np.int32)
            seqs = np.zeros((batch_size, max(lens)), dtype=np.int32)
            for i, form in enumerate(batch_forms):
                mask[i, :len(form)] = 1
                seqs[i, :len(form)] = form

            yield imgs, seqs, mask, sprase_labels


# Deprecated!! Queue Runners cannot be used as image is of variable size
class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """

    def __init__(self, batch_size=32, SEQ_LEN=50):
        self.dataX = tf.placeholder(
            dtype=tf.float32, shape=[None, 1, 128, 256])
        self.dataY = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.dataMask = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.queue = tf.RandomShuffleQueue(
            dtypes=[tf.float32, tf.int32, tf.int32],
            capacity=2000,
            min_after_dequeue=1000)

        self.SEQ_LEN = SEQ_LEN
        self.batch_size = batch_size

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many(
            [self.dataX, self.dataY, self.dataMask])

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch, mask_batch = self.queue.dequeue_many(
            self.batch_size)
        return images_batch, labels_batch, mask_batch

    def thread_main(self, sess):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        for dataX, dataY, dataMask in data_iterator(self.batch_size,
                                                    self.SEQ_LEN):
            sess.run(
                self.enqueue_op,
                feed_dict={
                    self.dataX: dataX,
                    self.dataY: dataY,
                    self.dataMask: dataMask
                })

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess, ))
            t.daemon = True  # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads
