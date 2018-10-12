#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-01 12:12:42
# Last Modified by: xiaofeng
# Last Modified time: 2018年04月25日17:02:28
'''
创建CRNN网络结构进行文本+字符识别
整体网络结构为CNN+RNN(ATTENTION)
'''
import datetime
import os
import sys
import time
from functools import reduce
from operator import mul
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import ctc_ops

import config as cfg
import data_loaders_ctc
import tfutils
import tfutils.network
import tfutils.network_ctc
import tfutils.ops
import tfutils.optimizer
# from network_modify import Network
from network_ctc import Embedding, Linear, multi_rnn
from tfutils.ops import im2latexAttention


slim = tf.contrib.slim

KEEP_DROUPOUT = 0.5
EMBEDING_DIM = 512
ENC_DIM = 256
DEC_DIM = ENC_DIM * 2
NUM_FEATS_START = 64
D = NUM_FEATS_START * 8
DIMS_INPUT = 512
DIMS_HIDDEN = 128
DIMS_OUTPUT = 512
VOCAB_SIZE = cfg.VOCABLARY_SIZE  # vocab size
print('vocab_size', VOCAB_SIZE)
H = 20
W = 50
NUM_LAYERS = 0
IMG_PATH = cfg.IMG_DATA_PATH
# PROPERTIES = cfg.PROPERTIES
PROPERTIES = cfg.DATA_ROOT + 'properties_ctc.npy'
ckpt_path = cfg.CHECKPOINT_PATH
property = np.load(PROPERTIES).tolist()


#  Create model
def main(net_name, learning_style, optimizer_name, num_layers):
    print('===============================================================')
    print('net name:{}--learning_style is:{}--optimizer_name:{}'.format(net_name,
                                                                        learning_style, optimizer_name))
    print('===============================================================')
    if net_name not in cfg.NET_LIST:
        print('net_name:{} is wrong!!!!!!')
        sys.exit()
    #===================================占位符=============================================#
    # 使用softmax损失函数
    X = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32, name='input_img')
    mask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='mask')
    seqs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='label')
    # 进行ctc损失函数占位符
    sprase_targets = tf.sparse_placeholder(dtype=tf.int32, name='sprase_labels')
    seq_length = tf.placeholder(shape=(None), dtype=tf.int32, name='sequence_length')

    # 设置模型的droupout
    keep_dropout = tf.placeholder(tf.float32)
    input_seqs = seqs[:, :-1]
    target_seqs = seqs[:, 1:]

    # ==============================sess设置========================================#

    config = tf.ConfigProto(
        device_count={"CPU": cfg.CPU_NUMS},
        intra_op_parallelism_threads=cfg.CPU_THREADS)
    # 如果存在GPU，进行gpu设置
    if cfg.GPU:
        config.gpu_options.per_process_gpu_memory_fraction = cfg.GPU_PERCENTAGE

    sess = tf.Session(config=config)

    # 进行模型整个权重保存位置的设置
    saver2_path = os.path.join(ckpt_path, net_name, learning_style)
    if not os.path.exists(saver2_path):
        os.makedirs(saver2_path)
    saved_filist = os.listdir(saver2_path)
    saver2_path_name = os.path.join(saver2_path, '%s_%s_trained.ckpt' % (net_name,
                                                                         learning_style))
    #==============================加载预训练参数=============================================#
    net_func = tfutils.network.net_fun(net_name=net_name, num_classes=None)
    ctx, _ = net_func(inputs=X)
    #==============================这里只加载网络中存在的参数====================================#
    # 如果不是第一次训练，那么就进行公开数据集的权重加载
    if not saved_filist:
        saver1 = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(cfg.PRETRAINED, net_name, '{}.ckpt'.format(net_name))
        print('load the pretrained of the net:{}, from:{}'.format(net_name, checkpoint_path))
        saver1.restore(sess, checkpoint_path)
    # =========resnet 之后连接卷积层，与LSTM连接，保证卷积层的输出通道数目============================#
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        biases_initializer=tf.zeros_initializer(), padding='SAME'):
        ctx = slim.conv2d(ctx, DIMS_INPUT, [1, 1], scope='CNN_OUT')
    # ==============================卷积层连接====================================================#
    # with tf.device('/cpu:0'):
    #     embedding = tf.Variable(
    #         tf.random_uniform([VOCAB_SIZE, EMBEDING_DIM], -1.0, 1.0),
    #         trainable=True,
    #         name='Embedding')
    #     embed = tf.nn.embedding_lookup(embedding, input_seqs)

    """ 添加attention的encode——decode """
    # seq2seq模型
    # 根据预设的层数进行RNN网络的选择
    if num_layers != 0:
        out = tfutils.ops.MutBigru(input=ctx, d_i=DIMS_INPUT, d_h=DIMS_HIDDEN,
                                   d_o=VOCAB_SIZE+1, name='MutBiGru', num_layer=num_layers)
    else:
        out = tfutils.ops.bilstm(input=ctx, d_i=DIMS_INPUT, d_h=DIMS_HIDDEN,
                                 d_o=VOCAB_SIZE+1, name='bilstm', trainable=True)
    # predict = tfutils.ops.lstm_fc(input=out, d_i=DIMS_OUTPUT,
    #                               d_o=VOCAB_SIZE, name='Logits', trainable=True)

    # ==========================使用ctc损失函数进行模型的优化迭代========================== #
    """ 定义ctc损失函数及根据函数名选择指定的优化器，并采用learning decay策略 """

    avg_loss = tf.reduce_mean(
        tf.nn.ctc_loss(
            labels=sprase_targets, inputs=out, sequence_length=seq_length))

    global_step = tf.train.create_global_step()

    #=======================================学习率设置====================================#
    learning_rate = tfutils.optimizer.configure_learning_rate(
        learning_rate_decay_type=learning_style,
        learning_rate=cfg.LEARNING_RATE,
        decay_steps=cfg.DECAY_STEPS,
        learning_rate_decay_rate=cfg.DECAY_RATE,
        global_step=global_step)
    #================================进行优化器更新并定义正确率评价指标=========================#
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tfutils.optimizer.configure_optimizer(
            optimizer_name=optimizer_name, learning_rate=learning_rate)

    gvs = optimizer.compute_gradients(avg_loss)
    capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    gradient_norms = [tf.norm(grad) for grad, var in gvs]

    with tf.name_scope("decode"):
        decoded, log_prob = ctc_ops.ctc_beam_search_decoder(
            out, seq_length, merge_repeated=False)

    with tf.name_scope("accuracy"):
        distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), sprase_targets)
        # 计算label error rate (accuracy)，该变量为模型正确率评价指标
        ler = tf.reduce_mean(distance, name='label_error_rate')

    tf.summary.scalar('larning_rate', learning_rate)
    tf.summary.scalar('ctc_loss', avg_loss)
    tf.summary.scalar('edit_distance_accuracy', ler)
    tf.summary.histogram('model_loss_his', avg_loss)
    tf.summary.histogram('model_acc_his', ler)
    merged = tf.summary.merge_all()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    #==============================训练参数和预训练参数的确定================================#

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
    saver2 = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=0.5)

    """ 根据网络名称和优化方法确定存储文件的位置 """

    saver2_path = os.path.join(ckpt_path, net_name, learning_style,
                               '%s_%s_trained.ckpt' % (net_name, learning_style))

    kpt = tf.train.latest_checkpoint(saver2_path)
    startepo = 0
    if kpt != None:
        saver2.restore(sess, kpt)
        ind = kpt.find("-")
        startepo = int(kpt[ind+1:])
        print(startepo)

    section = '\n{0:=^40}\n'
    print(section.format('Run training epoch'))

    summary_path = os.path.join(cfg.SUMMARY_PATH, net_name, learning_style)
    suammry_writer = tf.summary.FileWriter(
        summary_path, flush_secs=60, graph=sess.graph)

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_start = time.time()

    for epoch in range(cfg.EPOCH_NUMS):  # 样本集迭代次数
        epoch_start = time.time()
        if epoch < startepo:
            continue

        print("epoch start:", epoch, "total epochs= ", cfg.EPOCH_NUMS)

        train_cost = 0
        train_ler = 0
        iter_nums = 0
        # 生成训练数据，将每个epoch的数据作为一个批次，并且每次训练一个batch的样本
        iters_per_epoch = data_loaders_ctc.data_iterator('train', 32)
        for imgs, labels, mask, sprased in iters_per_epoch:
            iter_nums += 1
            ctx_out = sess.run(ctx, feed_dict={X: imgs})
            shape = np.shape(ctx_out)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            print('ctx_out', np.shape(ctx_out))
            lengths = np.asarray([H * W] * N, dtype=np.int64)

            rnn_out = sess.run(out, feed_dict={X: imgs})

            feed = {X: imgs, seqs: labels, sprase_targets: sprased,
                    seq_length: lengths, keep_dropout: KEEP_DROUPOUT}

            print('rnn_out', np.shape(rnn_out))
            print('imgs', np.shape(imgs))
            print('labels_details', labels)

            print('lengths', len(lengths), lengths)
            print('sprase:', sprased[2])

            batch_cost, _ = sess.run([avg_loss, train_step], feed_dict=feed)

            print('batch_cost', batch_cost)

            train_cost += batch_cost

            if (iter_nums + 1) % 1 == 0:
                print('loop:', iter_nums, 'Train cost: ', train_cost / (iter_nums + 1))
                feed2 = {X: imgs, seqs: labels, sprase_targets: sprased,
                         seq_length: lengths, keep_dropout: 1.0}

                d, train_ler, summary = sess.run([decoded[0], ler, merged], feed_dict=feed2)
                dense_decoded = tf.sparse_tensor_to_dense(
                    d, default_value=-1).eval(session=sess)
                dense_labels = data_loaders_ctc.sparse_tuple_to_texts_ch(sprase_targets)

                counter = 0
                print('Label err rate: ', train_ler)
                for orig, decoded_arr in zip(dense_labels, dense_decoded):
                    # convert to strings
                    decoded_str = data_loaders_ctc.ndarray_to_text_ch(decoded_arr)
                    print(' file {}'.format(counter))
                    print('Original: {}'.format(orig))
                    print('Decoded:  {}'.format(decoded_str))
                    counter = counter+1
                    break

        epoch_duration = time.time() - epoch_start

        log = 'Epoch {}/{}, train_cost: {:.3f}, train_acc: {:.3f}, time: {:.2f} sec'
        print(log.format(epoch, cfg.EPOCH_NUMS, train_cost, train_ler, epoch_duration))
        saver2.save(sess, saver2_path + "char_formula.cpkt", global_step=epoch)
        suammry_writer.add_summary(summary)
    train_duration = time.time() - train_start
    print('Training complete, total duration: {:.2f} min'.format(train_duration / 60))

    coord.request_stop()
    coord.join(thread)
    sess.close()


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
    print('num_layers of rnn modul:', NUM_LAYERS)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    main(
        net_name='resnet_v2_50',
        learning_style='exponential',
        optimizer_name='adam',
        num_layers=NUM_LAYERS)
