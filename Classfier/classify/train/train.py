#!/usr/bin/env python
# -*- coding: utf-8 -*-
_Author_ = 'xiaofeng'

import tensorflow as tf
import datetime
import os
import argparse
import config as cfg
from net.net_tiny_class import BUILDNET
from util.timer import Timer
from dataset.data_to_tfrecord import run_dataset_tfrecord

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net):
        self.net = net
        self.epoches = cfg.EPOCH_NUMS
        self.max_iter = cfg.ITER
        self.weight_file = cfg.WEIGHT_DIR
        self.meta_file = cfg.META_DIR
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.cfg_txt = os.path.join(cfg.TRAINING_PROCESS_CONFIG_DIR,
                                    datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        self.summary_dir = cfg.SUMMARY_SAVED
        self.output_dir = cfg.MODEL_OUTPUT_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.cfg_txt):
            os.makedirs(self.cfg_txt)
        self.save_cfg()

        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.summary_dir, flush_secs=60)

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # self.global_step = tf.cast(self.epoches * self.max_iter, tf.float32)
        # print('global_step', self.global_step)
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(
            self.net.total_loss, global_step=self.global_step)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        self.images, self.labels = run_dataset_tfrecord(is_training=True, shuffling=False)

        gpu_options = tf.GPUOptions()

        config = tf.ConfigProto(device_count={"CPU": 1},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=1,
                                intra_op_parallelism_threads=3,  # limit the threads
                                log_device_placement=False
                                )

        self.sess = tf.Session(config=config)
        initop = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(initop)
        # self.variable_to_restore = tf.global_variables()
        # self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=0.5)
        if self.weight_file is not None and self.meta_file is not None:
            print('Restoring weights from: ' + self.weight_file)
            self.saver = tf.train.import_meta_graph(self.meta_file)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(cfg.WEIGHT_DIR))

        # if self.weights_file is not None:
        #     print('Restoring weights from: ' + self.weights_file)
        #     self.restorer.restore(self.sess, self.weights_file)
        #
        # self.writer.add_graph(self.sess.graph)

    def train(self):
        train_timer = Timer()
        load_timer = Timer()
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        for epoch in range(self.epoches + 1):
            for step in range(1, self.max_iter + 1):
                load_timer.tic()
                # images, labels = run_dataset_tfrecord()
                image_batch, labels_batch = self.sess.run([self.images, self.labels])
                load_timer.toc()
                feed_dict = {self.net.images: image_batch, self.net.labels: labels_batch}
                if (step + self.max_iter * epoch) % self.summary_iter == 0:
                    if (step + self.max_iter * epoch) % (self.summary_iter * 10) == 0:
                        train_timer.tic()
                        summary_str, loss, acc, _ = self.sess.run(
                            [self.summary_op, self.net.total_loss, self.net.accurate, self.train_op],
                            feed_dict=feed_dict)
                        train_timer.toc()
                        log_str = ('{}Step: {}, Learning rate: {},'
                                   ' Loss: {:5.3f} Accuracy: {:5.3f} Speed: {:.3f}s/iter,'
                                   ' Load: {:.3f}s/iter, Remain: {}').format(
                            datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                            int(step + self.max_iter * epoch),
                            round(self.learning_rate.eval(session=self.sess), 6),
                            loss,
                            acc,
                            train_timer.average_time,
                            load_timer.average_time,
                            train_timer.remain((step + self.max_iter * epoch), self.epoches * self.max_iter))
                        print(log_str)

                    else:
                        train_timer.tic()
                        summary_str, _ = self.sess.run(
                            [self.summary_op, self.train_op],
                            feed_dict=feed_dict)
                        train_timer.toc()

                    self.writer.add_summary(summary_str, (step + self.max_iter * epoch))

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run([self.summary_op, self.train_op], feed_dict=feed_dict)
                    train_timer.toc()

                if (step + self.max_iter * epoch) % self.save_iter == 0:
                    print('{} Saving checkpoint file to: {}'.format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        self.output_dir))
                    self.saver.save(self.sess, self.ckpt_file,
                                    (step + self.max_iter * epoch))
        coord.request_stop()
        coord.join(thread)

    def save_cfg(self):

        with open(os.path.join(self.cfg_txt, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    # cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    # cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    # cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    # cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    # parser.add_argument('--threshold', default=0.2, type=float)
    # parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    # if args.data_dir != cfg.DATA_PATH:
    #     update_config_paths(args.data_dir, args.weights)

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    data = cfg.DATASET
    net = BUILDNET()

    solver = Solver(net)

    print('Start training ...')
    solver.train()
    # solver.coord.request_stop()
    # solver.coord.join(solver.thread)
    print('Done training.')


if __name__ == '__main__':
    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
