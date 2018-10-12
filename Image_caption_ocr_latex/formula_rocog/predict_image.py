#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-10 12:08:06
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-10 12:08:06
'''
对网络进行测试，输入图片输出对应的latex格式并进行可视化
'''
import datetime
import os
import random
import shutil
import sys
import time
from functools import reduce
from operator import mul
from subprocess import call
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import matplotlib as mpl
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python import pywrap_tensorflow

from config_formula import cfg as cfg
import data_loaders
import tflib
import tflib.network
import tflib.ops
import tflib.optimizer
from dataset.data_find_all_dirs import GetFileFromThisRootDir
from tflib.ops import im2latexAttention
from pprint import pprint
mpl.use('TkAgg')

pprint(cfg)

slim = tf.contrib.slim

# import re
NET_NAME = 'resnet_v2_50'
NUM_LAYERS = 3
LEARNING_STYLE = 'exponential'

BATCH_SIZE = 1  # 进行图像预测时，batch为1
RATIO = cfg.RATIO
SIZE = cfg.SIZE_LIST
LEARNING_DECAY = 20000
PREDICT_PATH = cfg.PREDICT_PATH
EMBEDING_DIM = 512
ENC_DIM = 256
DEC_DIM = ENC_DIM * 2
NUM_FEATS_START = 64
D = NUM_FEATS_START * 8
VOCAB_SIZE = cfg.VOCABLARY_SIZE  # vocab size
print('VOCAB_SIZE', VOCAB_SIZE)
H = 20
W = 50
IMG_PATH = cfg.IMG_DATA_PATH
PROPERTIES = cfg.PROPERTIES

ckpt_path = cfg.CHECKPOINT_PATH

BASIC_SKELETON = r'''
\documentclass[24pt,UTF8]{ctexart}
\usepackage{amsmath}
\usepackage{underscore}
\pagestyle{empty}

\begin{document}

%s

\end{document}
'''

RENDERING_SETUPS = {'basic': [
    BASIC_SKELETON, "convert -density 200 -quality 200 %s.pdf %s.png",
    lambda filename: os.path.isfile(filename + ".png")]}

DEVNULL = open(os.devnull, "w")


class Net():
    def __init__(self):
        self.net_name = 'resnet_v2_50'
        self.num_layers = 0
        self.learning_style = 'exponential'
        if self.net_name not in cfg.NET_LIST:
            print('net_name:{} is wrong!!!!!!')
            sys.exit()
        #===================================占位符=============================================#
        self.X = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32, name='input_img')
        self.seqs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='label')
        self.mask = tf.placeholder(shape=(None, None), dtype=tf.int32, name='mask')
        self.input_seqs = self.seqs[:, :-1]
        self.target_seqs = self.seqs[:, 1:]
        self.net_func = tflib.network.net_fun(net_name=self.net_name, num_classes=None)
        self.ctx, _ = self.net_func(inputs=self.X)
        self.D = 512
        self.H = 20
        self.W = 50
        #==============================这里只加载网络中存在的参数====================================#
        # 如果不是第一次训练，那么就进行公开数据集的权重加载
        # =========resnet 之后连接卷积层，与LSTM连接，保证卷积层的输出通道数目============================#
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.zeros_initializer(), padding='SAME'):
            self.ctx = slim.conv2d(self.ctx, cfg.MODEL.FEATURE, [3, 3], scope='CNN_OUT')

        """ embeding shape 为(batch,label_length,embeding_dim) """
        self.emb_seqs = tflib.ops.Embedding(name='Embedding', n_symbols=cfg.VOCABLARY_SIZE,
                                            output_dim=cfg.MODEL.DIMS_INPUT, indices=self.input_seqs)
        """
        添加attention的encode——decode
        seq2seq模型
        根据预设的层数进行RNN网络的选择
        输出为batch_size,label_length,dims_out
        对于存在attention层，dims_out=dims_attention
        """
        self.rnn_out, _ = tflib.ops.im2latexAttention(
            name='AttLSTM', inputs=self.emb_seqs, ctx=self.ctx, input_dim=cfg.MODEL.DIMS_INPUT,
            ENC_DIM=cfg.MODEL.DIMS_HIDDEN, DEC_DIM=cfg.MODEL.DIMS_ATTENTION, D=self.D, H=self.H, W=self.W)
        # 进行全连接输出维度转换，batch_size,label_length,dims_out变成维度batch_size,label_length,vocab_size
        self.logits = tflib.ops.Linear('MLP.1', self.rnn_out, cfg.MODEL.DIMS_ATTENTION, cfg.VOCABLARY_SIZE)
        """ 使用输入的字符预测当前最后一个字符 """
        self.output_index = tf.argmax(tf.nn.softmax(self.logits[:, -1]), axis=1)
        """ 使用输入的字符进行错位预测 """
        self.output_index_dislocation = tf.argmax(tf.reshape(self.logits, [-1, cfg.VOCABLARY_SIZE]), axis=1)

        # 将(batch_size,label_length,vocab_size)reshape成(batch_size*label_length,vocab_size)
        # output = tf.reshape(logits, [-1, cfg.VOCABLARY_SIZE])
        # 找到每一列的最大值，返回的维度为output——index为(batch_size*label_length)
        # output_index = tf.to_int32(tf.argmax(input=output, axis=1, name='argmax'))


def load_formula_model(ckpt_path, sess, graphic):
    with sess.as_default():
        with graphic.as_default():
            net = Net()
            saver_formula = tf.train.Saver(tf.global_variables())
            print('Restore the weight files from: {}'.format(ckpt_path))
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in var_to_shape_map:
                print("Tensor_name is : ", key)
            saver_formula.restore(sess, ckpt.model_checkpoint_path)
            print("Load formula recognise session done")
    return sess, saver_formula, net


# 加载存储的符号文件并转化成本list形式
properties = np.load(cfg.PROPERTIES).tolist()


def remove_temp_files(name):
    """ Removes .aux, .log, .pdf and .tex files for name """
    os.remove(name + ".aux")
    os.remove(name + ".log")
    os.remove(name + ".pdf")
    os.remove(name + ".tex")


def formula_to_image(formula, file_name, file_exten, formula_ori):
    """ Turns given formula into images based on RENDERING_SETUPS
    returns list of lists [[image_name, rendering_setup], ...], one list for
    each rendering.
    Return None if couldn't render the formula"""
    formula = formula.strip("%")
    skiping = []
    for rend_name, rend_setup in RENDERING_SETUPS.items():
        full_path = file_name + "_predict"
        latex = rend_setup[0] % formula
        if os.path.exists(full_path + '.png'):
            print('Render new name')
            full_path = file_name + '_predict_' + str(random.randint(0, 1000))
        # Write latex source
        with open(full_path + ".tex", "w") as f:
            f.write(latex)
        # Call pdflatex to turn .tex into .pdf
        code = call(["pdflatex", '-interaction=nonstopmode', '-halt-on-error',
                     full_path + ".tex"],
                    stdout=DEVNULL, stderr=DEVNULL)
        if code != 0:
            os.system("rm -rf " + full_path + "*")
            print('Image can not  generate: ', full_path + file_exten)
            with open(cfg.PREDICT_PATH_LOG + 'Not_generate_log_'+str(TEST_NUMS)+'.txt',
                      'a') as lob:
                wr = file_name + '   ' + formula_ori + '\n'
                lob.write(wr)
            return False

        # Turn .pdf to .png
        # Handles variable number of places to insert path.
        # i.e. "%s.tex" vs "%s.pdf %s.png"
        full_path_strings = rend_setup[1].count("%") * (full_path, )
        code = call((rend_setup[1] % full_path_strings).split(" "),
                    stdout=DEVNULL, stderr=DEVNULL)
        # Remove files
        try:
            remove_temp_files(full_path)
        except Exception as ex:
            # try-except in case one of the previous scripts removes these files
            # already
            with open(cfg.PREDICT_PATH + 'predict_error_'+str(TEST_NUMS)+'.txt', 'a') as tx:
                INFO_MESS = datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S') + '   ' + ex + '\n'
                tx.write(INFO_MESS)
            tx.close()
        print('Image was generated: ', cfg.PREDICT_PATH + file_name + '_predict.' + file_exten)
        return True


def score_target_pred(target, predict):
    class Found(Exception):
        pass

    length_target = len(target)
    length_pred = len(predict)
    count_pred, count_correct = 0, 0
    i = 0
    while i < length_target and count_pred < length_pred:
        value = target[i]
        pred = predict[count_pred]
        if value == pred:
            count_correct += 1
            count_pred += 1
            i += 1
        if value != pred:
            try:
                rest_ta = length_target - i
                rest_pred = length_pred - count_pred
                num_ta = min(rest_ta, 3)
                num_pred = min(rest_pred, 3)
                for i_plus in range(0, num_ta):
                    value_next = target[i + i_plus]
                    for count_pred_plus in range(0, num_pred):
                        pred_next = predict[count_pred + count_pred_plus]
                        if value_next == pred_next:
                            raise Found
            except Found:
                count_pred += count_pred_plus
                i += i_plus
            else:
                count_correct += 1
                i += 1
    correct = count_correct / len(target)
    return correct


def predict_img_latex(file, sess, net, target=None, Flage=False, calculate_falge=False):
    imgs = Image.open(file)
    width, height = imgs.size
    print('woidth,height', width, height)
    file_name = file.split('/')[-1].split('.')[0]
    file_exten = file.split('.')[-1]
    # If the data used is not the validate dataset
    if Flage:
        ratio = width / height
        ratio_diffreent = [np.abs(i - ratio) for i in cfg.RATIO]
        size_index = ratio_diffreent.index(min(ratio_diffreent))
        imgs = imgs.resize(cfg.SIZE_LIST[size_index], Image.LANCZOS)
        width, height = imgs.size
        print('The img tested resized was made and move to:{}'.format(cfg.PREDICT_PATH))
        shutil.copyfile(file, cfg.PREDICT_PATH + file_name + '_resized.' + file_exten)
    # The predictde length was baed on the img's width
    char_length = int(width / 2)
    # char_length = 300
    # Convert NCHW to NHWC
    imgs = np.asarray([np.asarray(imgs.convert('RGB'))], dtype=np.float32)
    print('img_shape', np.shape(imgs))
    inp_seqs = np.zeros((cfg.TEST.BATCH_SIZE, char_length)).astype('int32')
    inp_seqs[0, :] = properties['char_to_idx']['#START']
    # inp_seqs_dis = np.zeros((cfg.TEST.BATCH_SIZE, char_length)).astype('int32')
    # inp_seqs_dis[0, :] = properties['char_to_idx']['']
    tflib.ops.ctx_vector = []

    def idx_to_chars(Y): return ' '.join(map(lambda x: properties['idx_to_char'][x], Y))

    def idx_to_char2(y):
        out = ['' if i == 0 else properties['idx_to_char'][i] for i in y]
        # for i in y:
        #     c = '' if i == 0 else properties['idx_to_char'][i]
        #     out.append(c)
        return ' '.join(j for j in out)

    print('predict the latex')
    # inp_seqs = sess.run(output_index, feed_dict={X: imgs, seqs: inp_seqs})
    # 通过当前所有字符预测下一个字符"""  """
    for i in range(1, char_length):
        inp_seqs[:, i] = sess.run(
            [net.output_index], feed_dict={net.X: imgs, net.input_seqs: inp_seqs[:, :i]})
        # print('output_index_out', output_index_out)
        # print('output_index_dislocation_out', output_index_dislocation_out)
    str_ori = idx_to_chars(inp_seqs.flatten().tolist())
    formula = idx_to_chars(inp_seqs.flatten().tolist()).split('#END')[0].split('START')[-1]
    # formula = idx_to_char2(inp_seqs[0])
    # """ 进行错位预测 """
    # for i in range(1, char_length):
    #     if i + 1 <= char_length:
    #         inp_seqs_dis[:, 1:i+1] = sess.run([output_index_dislocation],
    #                                              feed_dict={X: imgs, input_seqs: inp_seqs_dis[:, :i]})
    # print('predict', idx_to_char2(inp_seqs_dis[0]))
    # # 预测出来的原始序列
    # formula_dis=idx_to_char2(inp_seqs_dis[0])
    # print('formula_dis', formula_dis)
    # correction calculated
    if not calculate_falge:
        assert target is None
        return formula
    else:
        predict_formula_idx = [int(properties['char_to_idx'][i]) for i in formula.split(' ') if list(i)]
        correct = score_target_pred(target, predict_formula_idx)
        print('correct', correct)
        # make img from the predict formula
        cureent_path = os.getcwd()
        with open(cfg.PREDICT_PATH_LOG + 'formula_predict'+str(TEST_NUMS)+'.txt', 'a') as log:
            wr = file_name + '   ' + formula + '   ' + str(correct) + '\n'
            log.write(wr)

        with open(cfg.PREDICT_PATH_LOG + 'formula_predict_ori'+str(TEST_NUMS)+'.txt', 'a') as ori:
            wr = file_name + '   ' + str_ori + '\n'
            ori.write(wr)

        # switch the directionary
        if os.path.exists(cfg.PREDICT_PATH):
            os.chdir(cfg.PREDICT_PATH)
        print('Generating img...')
        generate_or_not = formula_to_image(formula, file_name, file_exten, str_ori)
        print(correct, generate_or_not)
        if correct == 1.0 and generate_or_not:
            return 1
        elif 0.8 <= correct < 1.0 and generate_or_not:
            return 2
        elif correct < 0.8:
            return 3
        elif correct == 1.0 and not generate_or_not:
            return 4
        elif correct != 1.0 and not generate_or_not:
            return 5


def draw_result(result_list, test_num):
    n_groups = len(result_list)
    means = [100 * i for i in result_list]
    std = [10 * i for i in result_list]
    ind = np.arange(n_groups)
    width = 0.5
    labels = (u'correct_1', u'correct_81', u'correct_80', u'correct_1_N',
              u'Wrong_N')
    error_config = {'ecolor': '0.3'}
    plt.figure(dpi=300)
    result = plt.bar(
        ind, means, width, color='rgb', yerr=std, error_kw=error_config)
    plt.ylabel('Scores')
    plt.xlabel('Result')
    plt.title('Scores vs result')
    plt.xticks(ind + width / 2, labels)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            plt.text(
                rect.get_x() + rect.get_width() / 2.,
                1.05 * height,
                '%5.1f%%' % int(height),
                ha='center',
                va='bottom')

    autolabel(result)
    plt.savefig(cfg.PREDICT_PATH_LOG + 'predict_result' + str(test_num) + '.png')


TEST_IMG_PATH = cfg.DATA_ROOT + 'dataset_char_formula_enhance_train.ls'
# TEST_IMG_PATH = cfg.DATA_ROOT + 'dataset_char_formula_train.ls'
FORMULA_PATH = cfg.DATA_ROOT + 'new_char_formula_normal_enhance_out.ls'
TEST_NUMS = 10


def main(flage=True, draw_flag=False):
    graphic = tf.Graph()
    sess = tf.Session(graph=graphic)
    sess, _, net = load_formula_model(cfg.TEST.MODEL_SAVED, sess, graphic)
    if flage:
        if os.path.exists(cfg.IMG_DATA_PATH):
            file_list = GetFileFromThisRootDir([cfg.IMG_DATA_PATH],
                                               ['png', 'jpg'])
            # test_nums = int(0.1 * len(file_list))
            test_nums = 10
            random.shuffle(file_list)
            file_list = file_list[:test_nums]
            count = 1
            for file in file_list:
                print('-------{}/{}------'.format(count, test_nums))
                print('the file is:', file)
                formula_pre = predict_img_latex(file, sess, net)
                count += 1
                print('The formula is :', formula_pre)
            print('Complete')
        else:
            print(cfg.IMG_DATA_PATH, 'is wrong')
    else:
        assert os.path.exists(TEST_IMG_PATH)
        assert os.path.exists(FORMULA_PATH)
        index_value_list = open(TEST_IMG_PATH).read().split('\n')[:-1]
        formula_list = open(FORMULA_PATH).read().split('\n')[:-1]
        count = 0
        correct_1_generate, correct_81_generate, correct_80, correct_1_Ngenerate, wrong_Ngenerate = 0, 0, 0, 0, 0
        for i in index_value_list[:TEST_NUMS]:
            test_img_name = cfg.IMG_DATA_PATH + i.split(' ')[1]+'.png'
            test_img_index = int(i.split(' ')[0])
            target_formula = formula_list[test_img_index].split(' ')
            try:
                target_prob_index = [
                    int(properties['char_to_idx'][i]) for i in target_formula
                    if list(i)
                ]
                print('-------{}/{}------'.format(count + 1, TEST_NUMS))
                print('Input img is:', test_img_name)
                print('Input formula is:', formula_list[test_img_index])
                print('target_prob_index', target_prob_index)
                with open(cfg.PREDICT_PATH_LOG + 'formula_input_'+str(TEST_NUMS)+'.txt',
                          'a') as inp:
                    infor = test_img_name + '   ' + formula_list[test_img_index] + '\n'
                    inp.write(infor)
                result = predict_img_latex(test_img_name, target_prob_index,
                                           flage)
                print('result is：', result)
                count += 1
                if result == 1:
                    correct_1_generate += 1
                elif result == 2:
                    correct_81_generate += 1
                elif result == 3:
                    correct_80 += 1
                elif result == 4:
                    correct_1_Ngenerate += 1
                elif result == 5:
                    wrong_Ngenerate += 1
            except Exception as e:
                with open(cfg.PREDICT_PATH_LOG + 'err_log_'+str(TEST_NUMS)+'.txt', 'a') as err:
                    err.write(
                        i.split(' ')[0] + '   ' + str(e) + '   ' +
                        str(formula_list[test_img_index]) + '\n')
        print('count', count)
        C_1_G = correct_1_generate / count
        C_81_G = correct_81_generate / count
        C_80_G = correct_80 / count
        C_1_NG = correct_1_Ngenerate / count
        W_NG = wrong_Ngenerate / count
        with open(cfg.PREDICT_PATH_LOG+'result' + str(TEST_NUMS) + '.txt', 'w') as res:
            Info = 'Total count is:' + str(count) + '\n'\
                + 'correct_1_generate:' + str(C_1_G) + '   ' + str(correct_1_generate) + '\n'\
                + 'correct_81_generate:' + str(C_81_G) + '   ' + str(correct_81_generate) + '\n'\
                + 'correct_80:' + str(C_80_G) + '   ' + str(correct_80) + '\n'\
                + 'correct_1_Ngenerate:' + str(C_1_NG) + '   ' + str(correct_1_Ngenerate) + '\n'\
                + 'wrong_Ngenerate' + str(W_NG) + '   ' + str(wrong_Ngenerate) + '\n'
            print(Info)
            res.write(Info)
        if draw_flag:
            result_list = [C_1_G, C_81_G, C_80_G, C_1_NG, W_NG]
            draw_result(result_list, TEST_NUMS)

    sys.exit()


if __name__ == '__main__':
    # local img -flage=True,
    # dataset img -flage=False
    # if not os.path.exists(cfg.PREDICT_PATH_LOG):
    #     # shutil.rmtree(cfg.PREDICT_PATH_LOG)
    #     os.mkdir(cfg.PREDICT_PATH_LOG)
    # if not os.path.exists(cfg.PREDICT_PATH_LOG):
    #     os.mkdir(cfg.PREDICT_PATH_LOG)
    main(flage=True, draw_flag=True)
