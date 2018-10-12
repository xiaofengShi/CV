#!/usr/bin/env python
# -*- coding: utf-8 -*-
_Author_ = 'xiaofeng'

from net import net_tiny
import tensorflow as tf
import config as  cfg
import os, time
from dataset.data_to_tfrecord import run_dataset_tfrecord
from util.timer import Timer
from datetime import datetime
import numpy as np



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

slim = tf.contrib.slim

image_holder = tf.placeholder(tf.float32,
                              [cfg.BATCH_SIZE_TEST, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.IMAGE_CHANNEL])
label_holder = tf.placeholder(tf.int32, [cfg.BATCH_SIZE_TEST, cfg.CLASSES])

# image_batch, label_batch = run_dataset_tfrecord()
with  slim.arg_scope(net_tiny.tiny_net_arg_scope()):
    pred, end_point = net_tiny.tiny_net(inputs=image_holder, is_training=False)


def acc(logits, labels):
    count = 0
    flogits = []
    fgclasses = []
    for i in range(len(logits)):
        flogits.append(tf.reshape(logits[i], [-1, cfg.CLASSES]))
        fgclasses.append(tf.reshape(labels[i], [-1]))
        top_in_k = tf.nn.top_k(flogits, fgclasses, 1)

    return


# top_in_k = tf.nn.in_top_k(predictions=pred, targets=label_holder, k=1)
initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
ckpt_file = os.path.join(cfg.MODEL_OUTPUT_DIR, 'save.ckpt')

config = tf.ConfigProto(device_count={"CPU": 1},  # limit to num_cpu_core CPU usage
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=2,  # limit the threads
                        log_device_placement=False)
image_batch, label_batch = run_dataset_tfrecord(is_training=False, shuffling=False)
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=0.5)
with tf.Session(config=config) as sess:
    test = Timer()
    sess.run(initop)
    print('Start testing...')
    txt = open(os.path.join(cfg.TRAINING_PROCESS_CONFIG_DIR, 'test_process.txt'), 'w+')
    if cfg.WEIGHT_DIR is not None and cfg.META_DIR is not None:
        print('Restoring weights from: ' + cfg.WEIGHT_DIR)
        # saver = tf.train.import_meta_graph(cfg.META_DIR)
        saver.restore(sess, tf.train.latest_checkpoint(cfg.WEIGHT_DIR))
    else:
        print('Please confirm the weights dirs!!!!!!!!!')
        exit()
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    count = 0
    for iter in range(1, cfg.ITER_TEST + 1):
        start_time = time.time()
        image_feed_batch, label_feed_batch = sess.run([image_batch, label_batch])
        logits = sess.run(pred, feed_dict={image_holder: image_feed_batch, label_holder: label_feed_batch})
        # softmax_pred = sess.run(tf.nn.softmax(logits=logits))
        prediction_label = sess.run(tf.argmax(tf.nn.softmax(logits), 1))
        input_label = sess.run(tf.argmax(label_feed_batch, 1))
        for i in range(len(prediction_label)):
            if prediction_label[i] == input_label[i]:
                count += 1
        if iter % cfg.TEST_SHOW == 0:
            precision = count / (cfg.BATCH_SIZE_TEST * iter)
            remain = test.remain(iters=iter, max_iters=cfg.ITER)

            log = '\n' + '%s  Tested sample  num is : %s , total num is %s , precision  is : %s , ' \
                         ' remaining time is : %s ' % \
                  (datetime.now(), iter, cfg.ITER_TEST, precision, remain)
            print(log)
            txt.writelines(log)
coord.request_stop()
coord.join(thread)
print("testing finished!")
txt.close()
