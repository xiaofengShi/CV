#!/usr/bin/env python
# -*- coding: utf-8 -*-
_Author_ = 'xiaofeng'

'''
判断一张图像通过网络得到的分类类别名称
'''

import config as cfg
import os, re
from dataset.data_find_all_dirs import GetFileFromThisRootDir
import tensorflow as tf
from net import net_tiny
import matplotlib.pyplot as plt

slim = tf.contrib.slim


def modify_image(image_path):
    img_string = tf.read_file(image_path)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])
    img_resized = tf.reshape(img_resized, shape=[1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.IMAGE_CHANNEL])
    with  slim.arg_scope(net_tiny.tiny_net_arg_scope()):
        pred, end_point = net_tiny.tiny_net(inputs=img_resized, num_classes=cfg.CLASSES, is_training=False)
    prediction = tf.argmax(pred, 1)
    initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    config = tf.ConfigProto(device_count={"CPU": 1},  # limit to num_cpu_core CPU usage
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=2,  # limit the threads
                            log_device_placement=False)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(initop)
        print('Start testing...')
        if cfg.WEIGHT_DIR is not None and cfg.META_DIR is not None:
            print('Restoring weights from: ' + cfg.WEIGHT_DIR)
            saver.restore(sess, tf.train.latest_checkpoint(cfg.WEIGHT_DIR))
        else:
            print('Please confirm the weights dirs!!!!!!!!!')
            exit()
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)
        plt.imshow(img_decoded.eval())
        plt.title('Class recognized is:' + cfg.CLASSES_NAMES[sess.run(prediction)[0]])
        plt.show()

    coord.request_stop()
    coord.join(thread)
    print('Done')


def main():
    test_image_path = '1'
    modify_image(test_image_path)


if __name__ == '__main__':
    main()
