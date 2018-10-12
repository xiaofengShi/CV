#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-01 12:12:42
# Last Modified by: xiaofeng
# Last Modified time: 2018年04月25日17:02:28

import os
import shutil
import sys
sys.path.append(os.path.split(os.path.realpath(__file__))[0])
from PIL import Image
import tflib
import tflib.ops
import tflib.network
import tflib.optimizer
import data_loaders
import time
import datetime
import config as cfg
import numpy as np
from functools import reduce
from operator import mul
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

slim = tf.contrib.slim
# TRAIN
EMB_DIM = 80
ENC_DIM = 256
DEC_DIM = ENC_DIM * 2
NUM_FEATS_START = 64
D = NUM_FEATS_START * 8
V = cfg.V_OUT  # vocab size
H = 20
W = 50
IMG_PATH = cfg.IMG_DATA_PATH
PROPERTIES = cfg.PROPERTIES
ckpt_path = cfg.CHECKPOINT_PATH


#  Create model
def main(net_name, learning_style, optimizer_name):
    print('===============================================================')
    print('net name:{}--learning_style is:{}--optimizer_name:{}'.format(
        net_name, learning_style, optimizer_name))
    print('===============================================================')
    if net_name not in cfg.NET_LIST:
        print('net_name:{} is wrong!!!!!!')
        sys.exit()
    #===================================占位符=============================================#
    X = tf.placeholder(
        shape=(None, None, None, 3), dtype=tf.float32)  # resnet的占位符
    mask = tf.placeholder(shape=(None, None), dtype=tf.int32)
    seqs = tf.placeholder(shape=(None, None), dtype=tf.int32)
    input_seqs = seqs[:, :-1]
    target_seqs = seqs[:, 1:]

    #==============================sess设置================================================#
    config = tf.ConfigProto(
        device_count={"CPU": cfg.CPU_NUMS},
        intra_op_parallelism_threads=cfg.CPU_THREADS)
    # 如果存在GPU
    if cfg.GPU:
        config.gpu_options.per_process_gpu_memory_fraction = cfg.GPU_PERCENTAGE
    sess = tf.Session(config=config)
    #==============================加载预训练参数=============================================#
    net_fun = tflib.network.net_fun(net_name=net_name, num_classes=None)
    ctx, end_point = net_fun(inputs=X)
    #==============================这里只加载网络中存在的参数====================================#
    saver1 = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(cfg.PRETRAINED, net_name,
                                   '{}.ckpt'.format(net_name))
    print('load the pretrained of the net:{}, from:{}:'.format(
        net_name, checkpoint_path))
    saver1.restore(sess, checkpoint_path)
    # =========resnet 之后连接卷积层，与LSTM连接，保证卷积层的输出通道数目============================#
    with slim.arg_scope(
        [slim.conv2d],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(0.0005),
            biases_initializer=tf.zeros_initializer(),
            padding='SAME'):
        ctx = slim.conv2d(ctx, NUM_FEATS_START * 8, [3, 3], scope='CNN_OUT')
    #==============================卷积层连接====================================================#
    emb_seqs = tflib.ops.Embedding(name='Embedding', n_symbols=V, output_dim=EMB_DIM, indices=input_seqs)
    print('emb_seqs', np.shape(emb_seqs))
    out, state = tflib.ops.im2latexAttention('AttLSTM', emb_seqs, ctx, EMB_DIM,
                                             ENC_DIM, DEC_DIM, D, H, W)
    logits = tflib.ops.Linear('MLP.1', out, DEC_DIM, V)

    #==============================loss,predict==================================================#
    # predictions = tf.argmax(tf.nn.softmax(logits[:, -1]), axis=1)
    loss = tf.reshape(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=tf.reshape(logits, [-1, V]),
            labels=tf.reshape(seqs[:, 1:], [-1])), [tf.shape(X)[0], -1])
    # add paragraph ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓###
    output = tf.reshape(logits, [-1, V])
    output_index = tf.to_int32(tf.argmax(output, 1))
    true_labels = tf.reshape(seqs[:, 1:], [-1])
    correct_prediction = tf.equal(output_index, true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #### ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑###
    mask_mult = tf.to_float(mask[:, 1:])
    loss = tf.reduce_sum(loss * mask_mult) / tf.reduce_sum(mask_mult)

    global_step = tf.train.create_global_step()
    #=======================================学习率设置==============================================#
    learning_rate = tflib.optimizer.configure_learning_rate(
        learning_rate_decay_type=learning_style,
        learning_rate=cfg.LEARNING_RATE,
        decay_steps=cfg.DECAY_STEPS,
        learning_rate_decay_rate=cfg.DECAY_RATE,
        global_step=global_step)
    #====================================进行优化器更新===============================================#
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tflib.optimizer.configure_optimizer(
            optimizer_name=optimizer_name, learning_rate=learning_rate)
    #=====================================summary=================================================#
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    gradient_norms = [tf.norm(grad) for grad, var in gvs]
    tf.summary.scalar('larn_rate', learning_rate)
    tf.summary.scalar('model_loss', loss)
    tf.summary.scalar('model_accuracy', accuracy)
    tf.summary.histogram('model_loss_his', loss)
    tf.summary.histogram('model_acc_his', accuracy)
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    merged = tf.summary.merge_all()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    #==============================训练参数和预训练参数的确定==============================================#

    # 查看权重文件中的参数
    def how_many_paras(checkpoint_path):
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # for key in var_to_shape_map:
        #     print("tensor_name: ", key)
        #     print(reader.get_tensor(key))
        print('how many paras in the ckpt:', len(var_to_shape_map))

    # 统计模型参数量
    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params

    paras_nums = get_num_params()
    print('**************************************************************')
    print('total para name is:', len(tf.trainable_variables()))
    print('total para num is :', paras_nums)
    print('**************************************************************')

    sess.run(init)
    # restore the weights
    saver2 = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=0.5)
    saver2_name = os.path.join(
        ckpt_path, net_name, learning_style + '_' + optimizer_name,
        '%s_%s_%s_trained.ckpt' % (net_name, learning_style, optimizer_name))
    saver2_path = os.path.join(ckpt_path, net_name,
                               learning_style + '_' + optimizer_name)
    if not os.path.exists(saver2_path):
        os.makedirs(saver2_path)
    file_list = os.listdir(saver2_path)
    if file_list:
        for i in file_list:
            if i == 'checkpoint':
                print('======================================================')
                print('Restore the weight files form:', saver2_path)
                print('======================================================')
                # check how many paras in the ckpt
                # how_many_paras(saver2_name)
                saver2.restore(sess, tf.train.latest_checkpoint(saver2_path))
    summary_path = os.path.join(cfg.SUMMARY_PATH, net_name,
                                learning_style + '_' + optimizer_name)
    suammry_writer = tf.summary.FileWriter(summary_path, flush_secs=60, graph=sess.graph)

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    # function to predict the latex
    def score(set='valididate', batch_size=32):
        score_itr = data_loaders.data_iterator(set, batch_size)
        losses, pred = [], []
        for score_imgs, score_seqs, score_mask in score_itr:
            _loss, _acc = sess.run(
                [loss, accuracy],
                feed_dict={X: score_imgs, seqs: score_seqs, mask: score_mask
                           })
            losses.append(_loss)
            pred.append(_acc)
        set_loss = np.mean(losses)
        set_acc = np.mean(pred)
        perp = np.mean(list(map(lambda x: np.power(np.e, x), losses)))
        return set_loss, set_acc, perp

    times = []
    print("====================Compiled Train function!=====================")
    # Test is train func runs
    i = 0
    iter = 0
    best_perp = np.finfo(np.float32).max
    for i in range(i, cfg.EPOCH_NUMS):
        costs = []
        times = []
        pred = []
        itr = data_loaders.data_iterator('train', cfg.BATCH_SIZE)
        for train_img, train_seq, train_mask in itr:
            iter += 1
            start = time.time()
            _, _loss, _acc, summary = sess.run(
                [train_step, loss, accuracy, merged],
                feed_dict={
                    X: train_img,
                    seqs: train_seq,
                    mask: train_mask
                })
            times.append(time.time() - start)
            costs.append(_loss)
            pred.append(_acc)
            if iter % cfg.SAVED_NUMS == 0:
                print("Iter: %d (Epoch %d--%d)" % (iter, i + 1,
                                                   cfg.EPOCH_NUMS))
                print("\tMean cost: ", np.mean(costs))
                print("\tMean prediction: ", np.mean(pred))
                print("\tMean time: ", np.mean(times))
                print('\tSaveing summary to the path:', summary_path)
                print('\tSaveing model to the path:', saver2_name)
                suammry_writer.add_summary(summary)
                saver2.save(sess, saver2_name, global_step=global_step)
                print(
                    'learning rate and global step is:',
                    learning_rate.eval(session=sess),
                    global_step.eval(session=sess))

        print('=====================================================')
        print("\n\nEpoch %d Completed!" % (i + 1))
        print("\tMean train cost: ", np.mean(costs))
        print("\tMean train perplexity: ",
              np.mean(list(map(lambda x: np.power(np.e, x), costs))))
        print("\tMean time: ", np.mean(times))
        print('\n\n')
        print('processing the validate data...')
        val_loss, val_acc, val_perp = score('validate', cfg.BATCH_SIZE)
        print("\tMean val cost: ", val_loss)
        print("\tMean val acc: ", val_acc)
        print("\tMean val perplexity: ", val_perp)
        Info_out = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S') + '   ' + 'iter/epoch/epoch_nums-%d/%d/%d' % (
                iter, i, cfg.EPOCH_NUMS
        ) + '    ' + 'val cost/val perplexity:{}/{}'.format(
                val_loss, val_perp)
        with open(summary_path + 'val_loss.txt', 'a') as file:
            file.writelines(Info_out)
        file.close()
        if val_perp < best_perp:
            best_perp = val_perp
            saver2.save(sess, saver2_name)
            print("\tBest Perplexity Till Now! Saving state!")
    coord.request_stop()
    coord.join(thread)


if __name__ == '__main__':
    '''
    NET_LIST = ['vgg_16', 'vgg_19', 'resnet_v2_50', 'resnet_v2_152']
    LEARNING_STYLE = ['exponential', 'fixed', 'polynomial']    
    OPTIMIZER = ['adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'rmsprop', 'sgd']
    '''
    net_list = cfg.NET_LIST
    leraning_style = cfg.LEARNING_STYLE
    optimizer = cfg.OPTIMIZER
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('net_list:', net_list)
    print('leraning_style:', leraning_style)
    print('optimizer:', optimizer)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    main(
        net_name='resnet_v2_50',
        learning_style='exponential',
        optimizer_name='adam')
    '''
    sgd可以加载adam训练的权重，反之不行
    so:先使用adam训练，然后使用sgd进行训练
    '''
