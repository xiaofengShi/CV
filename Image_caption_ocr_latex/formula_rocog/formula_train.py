#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-01 12:12:42
# Last Modified by: xiaofeng
# Last Modified time: 2018年04月25日17:02:28
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
from tflib.network import net_fun
from tflib.ops import im2latexAttention

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


slim = tf.contrib.slim

EMB_DIM = 80
ENC_DIM = 256
DEC_DIM = ENC_DIM * 2
NUM_FEATS_START = 64
D = NUM_FEATS_START * 8
V = cfg.VOCABLARY_SIZE  # vocab size
H = 20
W = 50
property = np.load(cfg.PROPERTIES).tolist()


def idx_to_chars(Y): return ' '.join(map(lambda x: property['idx_to_char'][x], Y))


#  Create model
def main():
    print('='*40)
    print('Net name:{}'.format(cfg.SAVER_SUMMARY.NET_NAME_SELECT))
    print('Learning Style:{}'.format(cfg.SAVER_SUMMARY.LEARNING_STYLE_SELECT))
    print('Optimizer is:{}'.format(cfg.SAVER_SUMMARY.OPTIMIZER_SELECT))
    print('='*40)
    #===================================占位符=============================================#
    X = tf.placeholder(shape=(None, None, None, None), dtype=tf.float32, name='input_img')
    seqs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='label')
    mask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='mask')
    input_seqs = seqs[:, :-1]
    target_seqs = seqs[:, 1:]
    # ==============================sess设置================================================#
    config = tf.ConfigProto(device_count={"CPU": cfg.CPU_NUMS}, intra_op_parallelism_threads=cfg.CPU_THREADS)
    # 如果存在GPU，进行gpu设置
    if cfg.GPU:
        config.gpu_options.per_process_gpu_memory_fraction = cfg.GPU_PERCENTAGE

    sess = tf.Session(config=config)
    ctx = tflib.network.vgg16(X)
    # net_fun = tflib.network.net_fun(net_name=cfg.SAVER_SUMMARY.NET_NAME_SELECT, num_classes=None)
    # ctx, _ = net_fun(inputs=X)
    # with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
    #                     weights_regularizer=slim.l2_regularizer(0.0005),
    #                     biases_initializer=tf.zeros_initializer(), padding='SAME'):
    #     ctx = slim.conv2d(ctx, cfg.MODEL.FEATURE, [3, 3], scope='CNN_OUT')

    #==============================这里只加载网络中存在的参数====================================#
    # 如果不是第一次训练，那么就进行公开数据集的权重加载
    # if not os.listdir(cfg.SAVER_SUMMARY.SAVER_PATH):
    #     saver1 = tf.train.Saver(tf.global_variables())
    #     checkpoint_path = os.path.join(
    #         cfg.PRETRAINED, cfg.SAVER_SUMMARY.NET_NAME_SELECT, '{}.ckpt'.format(
    #             cfg.SAVER_SUMMARY.NET_NAME_SELECT))
    #     print('load the pretrained of the net:{}, from:{}'.format(
    #         cfg.SAVER_SUMMARY.NET_NAME_SELECT, checkpoint_path))
    #     saver1.restore(sess, checkpoint_path)

    # ==============================卷积层连接====================================================#
    """ embeding shape 为(batch,label_length,embeding_dim) """
    emb_seqs = tflib.ops.Embedding(
        name='Embedding', n_symbols=cfg.VOCABLARY_SIZE, output_dim=cfg.MODEL.DIMS_INPUT, indices=input_seqs)

    # if num_layers != 0:
    #     rnn_out, _ = tflib.ops.multi_rnn(
    #         name='AttLSTM', inputs=ctx, dims_input=cfg.MODEL.DIMS_INPUT,
    #         attention_decode=cfg.MODEL.DIMS_ATTENTION, dims_hidden=cfg.MODEL.DIMS_HIDDEN,
    #         dims_output=cfg.VOCABLARY_SIZE, embeding=emb_seqs, num_layers=num_layers,
    #         droupout_keep_prob_input=1.0, droupout_keep_prob_output=0.5, trainable=True)
    # else:
    #     rnn_out, _ = tflib.ops.im2latexAttention(
    #         name='AttLSTM', inputs=emb_seqs, ctx=ctx, input_dim=cfg.MODEL.DIMS_INPUT,
    #         ENC_DIM=cfg.MODEL.DIMS_HIDDEN, DEC_DIM=cfg.MODEL.DIMS_ATTENTION, D=D, H=H, W=W)
    #     # rnn_out, state = tflib.ops.im2latexAttention('AttLSTM', emb_seqs, ctx, EMB_DIM,
    #     #                                              ENC_DIM, DEC_DIM, D, H, W)
    rnn_out, _ = tflib.ops.im2latexAttention(
        name='AttLSTM', inputs=emb_seqs, ctx=ctx, input_dim=cfg.MODEL.DIMS_INPUT,
        ENC_DIM=cfg.MODEL.DIMS_HIDDEN, DEC_DIM=cfg.MODEL.DIMS_ATTENTION, D=D, H=H, W=W)

    # 进行全连接输出维度转换，batch_size,label_length,dims_out变成维度batch_size,label_length,vocab_size
    logits = tflib.ops.Linear('logits', rnn_out, cfg.MODEL.DIMS_ATTENTION, cfg.VOCABLARY_SIZE)

    output_index_test = tf.argmax(tf.nn.softmax(logits[:, -1]), axis=1)
    """ 模型的预测输出进行reshape，总共长度为batch_size*label_length """
    # 将(batch_size,label_length,vocab_size)reshape成(batch_size*label_length,vocab_size)
    output = tf.reshape(logits, [-1, cfg.VOCABLARY_SIZE])
    # 找到每一列的最大值，返回的维度为output——index为(batch_size*label_length)
    output_index = tf.to_int32(tf.argmax(input=output, axis=1, name='argmax'))
    # 将(batch_size,label_length)reshape成(batch_size*label_length)
    true_labels = tf.reshape(target_seqs, [-1])

    loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=output, labels=true_labels), [tf.shape(X)[0], -1])

    mask_mult = tf.to_float(mask[:, 1:])
    # loss_softmax = tf.reduce_mean(loss)
    loss_softmax = tf.reduce_sum(loss * mask_mult) / tf.reduce_sum(mask_mult)
    correct_prediction = tf.equal(output_index, true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # global_step = tf.train.create_global_step()
    global_step = tf.Variable(0, trainable=False)
    #=======================================学习率设置==============================================#
    learning_rate = tflib.optimizer.configure_learning_rate(
        learning_rate_decay_type=cfg.SAVER_SUMMARY.LEARNING_STYLE_SELECT,
        learning_rate=cfg.TRAIN.LEARNING_RATE, decay_steps=cfg.TRAIN.DECAY_STEPS,
        learning_rate_decay_rate=cfg.TRAIN.DECAY_RATE, global_step=global_step)
    #====================================进行优化器更新===============================================#
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tflib.optimizer.configure_optimizer(
            optimizer_name=cfg.SAVER_SUMMARY.OPTIMIZER_SELECT,
            learning_rate=learning_rate)
    #=====================================summary=================================================#
    gvs = optimizer.compute_gradients(loss_softmax)
    capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    gradient_norms = [tf.norm(grad) for grad, var in gvs]
    tf.summary.scalar('larn_rate', learning_rate)
    tf.summary.scalar('model_loss', loss_softmax)
    tf.summary.scalar('model_accuracy', accuracy)
    tf.summary.histogram('model_loss_his', loss_softmax)
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

    # function to predict the latex
    def score(set='valididate', batch_size=32):
        score_itr = data_loaders.data_iterator(set, batch_size)
        losses, pred = [], []
        iter = 0
        print('Val dataset...')
        for score_imgs, score_seqs, seq_mask in score_itr:
            iter += 1
            feed = {X: score_imgs, seqs: score_seqs, mask: seq_mask}
            _loss, _acc = sess.run([loss_softmax, accuracy], feed_dict=feed)
            losses.append(_loss)
            pred.append(_acc)
            if iter % cfg.TRAIN.DISPLAY_NUMS == 0:
                print("\tMean cost: ", np.mean(losses), '___', _loss)
                print("\tMean prediction: ", np.mean(pred), '___', _acc)
        set_loss = np.mean(losses)
        set_acc = np.mean(pred)
        perp = np.mean(list(map(lambda x: np.power(np.e, x), losses)))
        return set_loss, set_acc, perp

    def saved_weight(sess, iter, output_dir, saver):
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix + '_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(output_dir, filename)
        # save
        saver.save(sess, filename)
        calculate = False
        if calculate:
            print('='*60)
            print('Restore the weight files form:', cfg.SAVER_SUMMARY.SAVER_PATH)
            print('='*60)
            # check how many paras in the ckpt
            how_many_paras(filename)
        print('\tWrote weight to: {:s}'.format(filename))

    paras_nums = get_num_params()
    print('*'*60)
    print('total para name is:', len(tf.trainable_variables()))
    print('total para num is :', paras_nums)
    print('*'*60)

    sess.run(init)
    # restore the weights
    saver2 = tf.train.Saver(max_to_keep=3, write_version=tf.train.SaverDef.V2)
    restore_iter = 0
    if os.listdir(cfg.SAVER_SUMMARY.SAVER_PATH):
        ckpt = tf.train.get_checkpoint_state(cfg.SAVER_SUMMARY.SAVER_PATH)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')

        saver2.restore(sess, ckpt.model_checkpoint_path)
        stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
        restore_iter = int(stem.split('_')[-1])
        sess.run(global_step.assign(restore_iter))
        print('done')
    summary_path = cfg.SAVER_SUMMARY.SUMMARY_PTH
    suammry_writer = tf.summary.FileWriter(logdir=summary_path, flush_secs=2, graph=tf.get_default_graph())

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('='*30+"Compiled Train function!"+'='*30)
    # Test is train func runs
    i = 0
    best_perp = np.finfo(np.float32).max
    print('restore_iter, max_iters', restore_iter, cfg.TRAIN.MAX_ITERS)
    iter = restore_iter
    while iter <= cfg.TRAIN.MAX_ITERS:
        if iter != 0:
            print(learning_rate.eval(session=sess))
        costs, times, pred = [], [], []
        itr = data_loaders.data_iterator('train', cfg.TRAIN.BATCH_SIZE)
        print('~'*30+"NEW_EPOCH"+'~'*30)
        for train_img, train_seq, train_mask in itr:
            iter += 1
            start = time.time()
            feed = {X: train_img, seqs: train_seq, mask: train_mask}
            _, _loss, _acc, summary = sess.run([train_step, loss_softmax, accuracy, merged],
                                               feed_dict=feed)
            suammry_writer.add_summary(summary=summary, global_step=global_step.eval(session=sess))
            times.append(time.time() - start)
            costs.append(_loss)
            pred.append(_acc)
            if iter % cfg.TRAIN.SAVED_NUMS == 0:
                print("Iter: %d, Max_iters: %d)" % (iter, cfg.TRAIN.MAX_ITERS))
                print("\tMean cost: ", np.mean(costs), '___', _loss)
                print("\tMean prediction: ", np.mean(pred), '___', _acc)
                print("\tMean time: ", np.mean(times))
                saved_weight(sess, iter, cfg.SAVER_SUMMARY.SAVER_PATH, saver2)
                # saver2.save(sess, saver2_path_name)
            if iter % cfg.TRAIN.DISPLAY_NUMS == 0:
                true_char = idx_to_chars(train_seq[0, 1:].flatten().tolist())
                predict_index, _ = sess.run([output_index, train_step],
                                            feed_dict={X: [train_img[0]],
                                                       seqs: [train_seq[0]],
                                                       mask: train_mask})
                predict_char = idx_to_chars(predict_index.flatten().tolist())
                print('\nLearning rate and global step is:', learning_rate.eval(
                    session=sess), global_step.eval(session=sess))
                print('\tTrue char is:', true_char)
                print('\tPredict char is:', predict_char)
            if iter % cfg.TRAIN.EVALUATE == 0:
                char_length = int(np.shape(train_img[0])[1] / 2)
                inp_seqs = np.zeros((cfg.TEST.BATCH_SIZE, char_length)).astype('int32')
                inp_seqs[0, :] = property['char_to_idx']['#START']
                tflib.ops.ctx_vector = []
                true_char = idx_to_chars(train_seq[0, 1:].flatten().tolist())
                for i in range(1, char_length):
                    inp_seqs[:, i] = sess.run(
                        [output_index_test], feed_dict={X: [train_img[0]], input_seqs: inp_seqs[:, :i]})
                formula_pred = idx_to_chars(inp_seqs.flatten().tolist()).split('#END')[0].split('START')[-1]
                print('True char is :', true_char)
                print('Predict char is:', formula_pred)
        print('='*30+'One Epoch Completed'+'='*30)
        print("\n\nEpoch %d Completed!" % (iter + 1))
        print("\tMean train cost: ", np.mean(costs))
        print("\tMean train perplexity: ",
              np.mean(list(map(lambda x: np.power(np.e, x), costs))))
        print("\tMean time: ", np.mean(times))
        print('\n\n')
        print('processing the validate data...')
        val_loss, val_acc, val_perp = score('validate', cfg.TRAIN.BATCH_SIZE)
        print("\tMean val cost: ", val_loss)
        print("\tMean val acc: ", val_acc)
        print("\tMean val perplexity: ", val_perp)
        Info_out = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '   ' + 'iter/max_iters-%d/%d' % \
            (iter, cfg.TRAIN.MAX_ITERS) + '    ' + 'val cost/val perplexity:{}/{}'.format(val_loss, val_perp)
        with open(summary_path + 'val_loss.txt', 'a') as file:
            file.writelines(Info_out)
        if val_perp < best_perp:
            best_perp = val_perp
            saved_weight(sess, iter, cfg.SAVER_SUMMARY.SAVER_PATH, saver2)
            print("\tBest Perplexity Till Now! Saving state!")
    coord.request_stop()
    coord.join(thread)


if __name__ == '__main__':
    '''
    NET_LIST = ['vgg_16', 'vgg_19', 'resnet_v2_50', 'resnet_v2_152']
    LEARNING_STYLE = ['exponentiaprintl', 'fixed', 'polynomial']
    OPTIMIZER = ['adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'rmsprop', 'sgd']
    '''
    print('~'*90)
    pprint(cfg)
    print('~' * 90)
    main()
