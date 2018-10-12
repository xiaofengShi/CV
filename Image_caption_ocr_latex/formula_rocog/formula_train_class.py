'''
File: formula_train_class.py
Project: formula_rocog
File Created: Friday, 29th June 2018 6:32:57 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Tuesday, 3rd July 2018 11:45:23 am
Modified By: xiaofeng (sxf1052566766@163.com)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''

'''
创建CRNN网络结构进行文本+字符识别
整体网络结构为CNN+RNN(ATTENTION)
2018年05月28日15:33:48：input，targets
'''
import datetime
import os
import sys
import time
from functools import reduce
from operator import mul
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import data_loaders
import tflib
import tflib.ops
import tflib.optimizer
from config_formula import cfg as cfg
from tflib.network_train_test import NET_TRAIN

slim = tf.contrib.slim


class Net_model(object):
    def __init__(self, sess, net):
        self.sess = sess
        self.net = net
        self.property = np.load(cfg.PROPERTIES).tolist()
        self.saver = tf.train.Saver(max_to_keep=3, write_version=tf.train.SaverDef.V2)
        self.summary_path = cfg.SUMMARY_PATH
        self.suammry_writer = tf.summary.FileWriter(
            logdir=self.summary_path, flush_secs=2, graph=tf.get_default_graph())

    def how_many_paras(self, checkpoint_path):
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # for key in var_to_shape_map:
        #     print("tensor_name: ", key)
        #     print(reader.get_tensor(key))
        print('how many paras in the ckpt:', len(var_to_shape_map))

    def get_num_params(self):
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    def idx_to_chars(self, Y):
        return ' '.join(map(lambda x: self.property['idx_to_char'][x], Y))
    # function to predict the latex

    def score(self, sess, set='valididate',  batch_size=32):
        score_itr = data_loaders.data_iterator(set, batch_size)
        losses, pred = [], []
        iters = 0
        print('Val dataset...')
        for score_imgs, score_seqs, seq_mask in score_itr:
            iters += 1
            feed = {self.net.img: score_imgs, self.net.seqs: score_seqs, self.net.mask: seq_mask}
            _loss, _acc = sess.run([self.net.loss_total, self.net.accuracy], feed_dict=feed)
            losses.append(_loss)
            pred.append(_acc)
            if iters % cfg.TRAIN.DISPLAY_NUMS == 0:
                print("\tMean cost: ", np.mean(losses), '___', _loss)
                print("\tMean prediction: ", np.mean(pred), '___', _acc)
        set_loss = np.mean(losses)
        set_acc = np.mean(pred)
        perp = np.mean(list(map(lambda x: np.power(np.e, x), losses)))
        return set_loss, set_acc, perp

    def saved_weight(self, sess, iters, output_dir):
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix + '_iter_{:d}'.format(iters + 1) + '.ckpt')
        filename = os.path.join(output_dir, filename)
        self.saver.save(sess, filename)
        calculate_params = False
        if calculate_params:
            print('='*60)
            print('Restore the weight files form:', cfg.CHECKPOINT_PATH)
            print('='*60)
            # check how many paras in the ckpt
            self.how_many_paras(filename)
        print('\tWrote weight to: {:s}'.format(filename))

    def train(self, sess):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tflib.optimizer.configure_learning_rate(
            learning_rate_decay_type=cfg.SAVER_SUMMARY.LEARNING_STYLE_SELECT,
            learning_rate=cfg.TRAIN.LEARNING_RATE, decay_steps=cfg.TRAIN.DECAY_STEPS,
            learning_rate_decay_rate=cfg.TRAIN.DECAY_RATE, global_step=global_step)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        optimizer = tflib.optimizer.configure_optimizer(
            optimizer_name=cfg.SAVER_SUMMARY.OPTIMIZER_SELECT, learning_rate=learning_rate)
        gvs = optimizer.compute_gradients(self.net.loss_total)
        capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
        train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        gradient_norms = [tf.norm(grad) for grad, var in gvs]
        tf.summary.scalar('larn_rate', learning_rate)
        tf.summary.scalar('model_loss', self.net.loss_total)
        tf.summary.scalar('model_accuracy', self.net.accuracy)
        tf.summary.histogram('model_loss_his', self.net.loss_total)
        tf.summary.histogram('model_acc_his', self.net.accuracy)
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
        merged = tf.summary.merge_all()

        paras_nums = self.get_num_params()
        print('*'*60)
        print('total para name is:', len(tf.trainable_variables()))
        print('total para num is :', paras_nums)
        print('*' * 60)
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        sess.run(init)
        # restore the weights
        restore_iter = 0
        if os.listdir(cfg.CHECKPOINT_PATH):
            ckpt = tf.train.get_checkpoint_state(cfg.CHECKPOINT_PATH)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
            restore_iter = int(stem.split('_')[-1])
            sess.run(global_step.assign(restore_iter))
            print('done')

        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('='*30+"Compiled Train function!"+'='*30)

        best_perp = np.finfo(np.float32).max
        print('restore_iter, max_iters', restore_iter, cfg.TRAIN.MAX_ITERS)

        iters = restore_iter
        while iters <= cfg.TRAIN.MAX_ITERS:
            if iters != 0:
                # print(learning_rate.eval(session=sess))
                times = int(iters/cfg.TRAIN.LERANING_DACEY)
                sess.run(tf.assign(learning_rate, learning_rate.eval() * cfg.TRAIN.GAMMA**(times)))
                print('current learning rate is:', learning_rate.eval())
            costs, times, pred = [], [], []
            itr = data_loaders.data_iterator('train', cfg.TRAIN.BATCH_SIZE)
            print('~'*30+"NEW_EPOCH"+'~'*30)
            for train_img, train_seq, train_mask in itr:
                iters += 1
                start = time.time()
                feed = {self.net.img: train_img, self.net.seqs: train_seq, self.net.mask: train_mask}
                _, _loss, _acc, summary = sess.run(
                    [train_step, self.net.loss_total, self.net.accuracy, merged],
                    feed_dict=feed)
                self.suammry_writer.add_summary(summary=summary, global_step=global_step.eval(session=sess))
                # print('_mask_mult:', tf.reduce_sum(_loss_ori * _mask_mult).eval(session=sess),
                #       tf.reduce_sum(_mask_mult).eval(session=sess))
                times.append(time.time() - start)
                costs.append(_loss)
                pred.append(_acc)
                if iters % cfg.TRAIN.SAVED_NUMS == 0:
                    print("Iter: %d, Max_iters: %d)" % (iters, cfg.TRAIN.MAX_ITERS))
                    print("\tMean cost: ", np.mean(costs), '___', _loss)
                    print("\tMean prediction: ", np.mean(pred), '___', _acc)
                    print("\tMean time: ", np.mean(times))
                    self.saved_weight(sess, iters, cfg.CHECKPOINT_PATH)

                    # saver2.save(sess, saver2_path_name)
                if iters % cfg.TRAIN.DISPLAY_NUMS == 0:
                    print('-' * 50)
                    print('Display the prediction')
                    true_char = self.idx_to_chars(train_seq[0, 1:].flatten().tolist())
                    feed = {
                        self.net.img: [train_img[0]],
                        self.net.seqs: [train_seq[0]],
                        self.net.mask: [train_mask[0]]}
                    predict_index = sess.run(self.net.output_index, feed_dict=feed)
                    predict_char = self.idx_to_chars(predict_index.flatten().tolist())
                    print('\nLearning rate and global step is:', learning_rate.eval(
                        session=sess), global_step.eval(session=sess))
                    print('\tTrue char is:', true_char)
                    print('\tPredict char is:', predict_char)
                if iters % cfg.TRAIN.EVALUATE == 0:
                    print('=' * 50)
                    print('Evaluate the model')
                    char_length = int(np.shape(train_img[0])[1] / 2)
                    inp_seqs = np.zeros((cfg.TEST.BATCH_SIZE, char_length)).astype('int32')
                    # inp_seqs[0, :] = self.property['char_to_idx']['#START']
                    inp_seqs[:, 0] = self.property['char_to_idx']['#START']
                    tflib.ops.ctx_vector = []
                    true_char = self.idx_to_chars(train_seq[0].flatten().tolist())
                    for i in range(1, char_length):
                        feed = {self.net.img: [train_img[0]], self.net.input_seqs: inp_seqs[:, :i]}
                        inp_seqs[:, i] = sess.run(self.net.prediction, feed_dict=feed)
                    formula_pred = self.idx_to_chars(inp_seqs.flatten().tolist()).split('#END')[
                        0].split('#START')[-1]
                    print('\tTrue char is :', true_char)
                    print('\tPredict char is:', formula_pred)
            print('='*30+'One Epoch Completed'+'='*30)
            print("\n\nEpoch %d Completed!" % (iters + 1))
            print("\tMean train cost: ", np.mean(costs))
            print("\tMean train perplexity:",
                  np.mean(list(map(lambda x: np.power(np.e, x), costs))))
            print("\tMean time: ", np.mean(times))
            print('\n\n')
            print('processing the validate data...')
            val_loss, val_acc, val_perp = self.score(
                sess=sess, set='validate', batch_size=cfg.TRAIN.BATCH_SIZE)
            print("\tMean val cost: ", val_loss)
            print("\tMean val acc: ", val_acc)
            print("\tMean val perplexity: ", val_perp)
            Info_out = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '   ' + 'iters/max_iters-%d/%d' % \
                (iters, cfg.TRAIN.MAX_ITERS) + '    ' + 'val cost/val perplexity:{}/{}'.format(val_loss, val_perp)
            with open(self.summary_path + 'val_loss.txt', 'a') as file:
                file.writelines(Info_out)
            if val_perp < best_perp:
                best_perp = val_perp
                self.saved_weight(sess, iters, cfg.CHECKPOINT_PATH)
                print("\tBest Perplexity Till Now! Saving state!")
        coord.request_stop()
        coord.join(thread)


def train_net(net_model):
    print('~'*90)
    pprint(cfg)
    print('~' * 90)
    if net_model == 'train':
        network = NET_TRAIN()
    else:
        raise 'Must input the string: train'
    config = tf.ConfigProto(device_count={"CPU": cfg.CPU_NUMS}, intra_op_parallelism_threads=cfg.CPU_THREADS)
    if cfg.GPU:
        config.gpu_options.per_process_gpu_memory_fraction = cfg.GPU_PERCENTAGE
    with tf.Session(config=config) as sess:
        model = Net_model(sess, network)
        print('Solving')
        model.train(sess)
        print('Done training')


if __name__ == '__main__':
    train_net('train')
