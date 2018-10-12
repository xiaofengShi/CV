#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-10 12:08:06
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-10 12:08:06
'''
原始仓库网络结构进行测试
'''
from PIL import Image
import tensorflow as tf
import os
import sys
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
# import tflib
import tflib.ops
import datetime
import tflib.network
from tqdm import tqdm
import numpy as np
import random
import data_loaders
from subprocess import call
from dataset.data_find_all_dirs import GetFileFromThisRootDir
import os
import cv2
from io import StringIO
import config as cfg
import shutil
# import re

BATCH_SIZE = 1  # 进行图像预测时，batch为1
EMB_DIM = 80
ENC_DIM = 256
PRECEPTION = 0.6
THREAD = 3
DEC_DIM = ENC_DIM * 2
NUM_FEATS_START = 64
D = NUM_FEATS_START * 8
V = 134  # vocab size
NB_EPOCHS = 100000
H = 20
W = 50
RATIO = cfg.RATIO
SIZE = cfg.SIZE_LIST
LEARNING_DECAY = 20000
IMG_PATH = cfg.IMG_DATA_PATH
PREDICT_PATH = cfg.PREDICT_PATH
PROPERTIES = cfg.PROPERTIES
# ckpt_path = cfg.CHECKPOINT_PATH
ckpt_path = '/Users/xiaofeng/Desktop/model_generate/ckpt'
summary_path = cfg.SUMMARY_PATH
BASIC_SKELETON = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\begin{document}

\begin{displaymath}
%s
\end{displaymath}

\end{document}
"""

RENDERING_SETUPS = {
    'basic': [
        BASIC_SKELETON, "convert -density 200 -quality 100 %s.pdf %s.png",
        lambda filename: os.path.isfile(filename + ".png")
    ]
}
DEVNULL = open(os.devnull, "w")

# build the model
X = tf.placeholder(shape=(None, None, None, None), dtype=tf.float32)
mask = tf.placeholder(shape=(None, None), dtype=tf.int32)
seqs = tf.placeholder(shape=(None, None), dtype=tf.int32)
input_seqs = seqs[:, :-1]
target_seqs = seqs[:, 1:]
# Embedding
emb_seqs = tflib.ops.Embedding('Embedding', V, EMB_DIM, input_seqs)

ctx = tflib.network.im2latex_cnn(X, NUM_FEATS_START, True)
out, state = tflib.ops.im2latexAttention('AttLSTM', emb_seqs, ctx, EMB_DIM,
                                         ENC_DIM, DEC_DIM, D, H, W)
logits = tflib.ops.Linear(
    name='MLP.1', inputs=out, input_dim=DEC_DIM, output_dim=V)
# 设置输出的阈值，查看输出的结果
out_predict = tf.nn.softmax(logits[:, -1], axis=1)
# 查看输出结果中最后的一行，对应输出的类别
predictions = tf.argmax(tf.nn.softmax(logits[:, -1]), axis=1)

# 进行config配置
config = tf.ConfigProto(intra_op_parallelism_threads=THREAD)
config.gpu_options.per_process_gpu_memory_fraction = PRECEPTION
sess = tf.Session(config=config)
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
sess.run(init)
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=0.5)
print('Restore the weight files from: {}'.format(ckpt_path))

saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
coord = tf.train.Coordinator()
thread = tf.train.start_queue_runners(sess=sess, coord=coord)

# 加载存储的符号文件并转化成本list形式
# properties = np.load(PROPERTIES).tolist()
properties = np.load('/Users/xiaofeng/Desktop/properties_generate.npy').tolist()


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
        code = call(
            [
                "pdflatex", '-interaction=nonstopmode', '-halt-on-error',
                full_path + ".tex"
            ],
            stdout=DEVNULL,
            stderr=DEVNULL)
        if code != 0:
            os.system("rm -rf " + full_path + "*")
            print('Image can not  generate: ', full_path + file_exten)
            with open(cfg.PREDICT_PATH_LOG + 'Not_generate_log.txt',
                      'a') as lob:
                wr = file_name + '   ' + formula_ori + '\n'
                lob.write(wr)
            return False

        # Turn .pdf to .png
        # Handles variable number of places to insert path.
        # i.e. "%s.tex" vs "%s.pdf %s.png"
        full_path_strings = rend_setup[1].count("%") * (full_path, )
        code = call(
            (rend_setup[1] % full_path_strings).split(" "),
            stdout=DEVNULL,
            stderr=DEVNULL)
        # Remove files
        try:
            remove_temp_files(full_path)
        except Exception as e:
            # try-except in case one of the previous scripts removes these files
            # already
            with open(cfg.PREDICT_PATH_LOG + 'predict_error.txt', 'a') as tx:
                INFO_MESS = datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S') + '   ' + e + '\n'
                tx.write(INFO_MESS)
            tx.close()
        print('Image was generated: ',
              cfg.PREDICT_PATH + file_name + '_predict.' + file_exten)
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


def predict_img_latex(file, target, Flage=False, calculate_flage=False):
    imgs = Image.open(file)
    width, height = imgs.size
    file_name = file.split('/')[-1].split('.')[0]
    file_exten = file.split('.')[-1]
    # If the data used is not the validate dataset
    if Flage:
        ratio = width / height
        ratio_diffreent = [np.abs(i - ratio) for i in RATIO]
        size_index = ratio_diffreent.index(min(ratio_diffreent))
        imgs = imgs.resize(SIZE[size_index], Image.LANCZOS)
        width, height = imgs.size
        print('The img tested resized was made and move to:{}'.format(
            cfg.PREDICT_PATH))
        shutil.copyfile(
            file, cfg.PREDICT_PATH + file_name + '_resized.' + file_exten)
    imgs = np.asarray(imgs.convert('YCbCr'))[:, :, 0][None, None, :]
    # imgs = np.asarray(imgs.convert('RGB'))
    print('shape', np.shape(imgs))
    # The predictde length was baed on the img's width
    char_length = int(width / 2)
    # Convert NCHW to NHWC
    imgs = np.asarray(imgs, dtype=np.float32).transpose(0, 2, 3, 1)
    # imgs = np.asarray([np.asarray(imgs)], dtype=np.float32)
    inp_seqs = np.zeros((BATCH_SIZE, char_length)).astype('int32')
    inp_seqs[0, :] = properties['char_to_idx']['#START']
    tflib.ops.ctx_vector = []
    # displayPreds = lambda Y: display(Math(Y.split('#END')[0]))

    def idx_to_chars(Y): return ' '.join(map(lambda x: properties['idx_to_char'][x], Y))
    # predict the latex
    print('predict the latex')
    for i in range(1, char_length):
        inp_seqs[:, i], pre = sess.run(
            (predictions, out_predict),
            feed_dict={
                X: imgs,
                input_seqs: inp_seqs[:, :i]
            })
    # str_ori = idx_to_chars(inp_seqs[0, :])
    str_ori = idx_to_chars(inp_seqs.flatten().tolist())
    formula = idx_to_chars(
        inp_seqs.flatten().tolist()).split('#END')[0].split('START')[-1]
    if not calculate_flage:
        assert target is None
        return formula
    else:
        # correction calculated
        predict_formula_idx = [
            int(properties['char_to_idx'][i]) for i in formula.split(' ')
            if list(i)
        ]
        correct = score_target_pred(target, predict_formula_idx)
        print('correct', correct)
        # make img from the predict formula
        cureent_path = os.getcwd()
        with open(cfg.PREDICT_PATH_LOG + 'formula_latex_log.txt', 'a') as log:
            wr = file_name + '   ' + formula + '   ' + str(correct) + '\n'
            log.write(wr)
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
    plt.savefig(
        cfg.PREDICT_PATH_LOG + 'predict_result' + str(test_num) + '.png')


TEST_IMG_PATH = cfg.DATA_ROOT + 'validate_filter.lst'
FORMULA_PATH = cfg.DATA_ROOT + 'formulas.norm.lst'
TEST_NUMS = 10


def main(flage=True, draw_flag=False):
    if flage:
        if os.path.exists(cfg.IMG_DATA_PATH):
            file_list = GetFileFromThisRootDir([cfg.IMG_DATA_PATH],
                                               ['png', 'jpg'])
            # test_nums = int(0.1 * len(file_list))
            random.seed(122)
            random.shuffle(file_list)
            file_list = file_list[:TEST_NUMS]
            count = 1
            print(file_list)
            for file in file_list:
                print('-------{}/{}------:'.format(count, len(file_list)))
                print('file is:', file)
                result = predict_img_latex(file, None)
                count += 1
                print('predict formula is:', result)
            print('Complete')
        else:
            print(cfg.IMG_TEST_PATH, 'is wrong')
    else:
        assert os.path.exists(TEST_IMG_PATH)
        assert os.path.exists(FORMULA_PATH)
        index_value_list = open(TEST_IMG_PATH).read().split('\n')[:-1]
        formula_list = open(FORMULA_PATH).read().split('\n')[:-1]
        count = 0
        correct_1_generate, correct_81_generate, correct_80, correct_1_Ngenerate, wrong_Ngenerate = 0, 0, 0, 0, 0
        for i in index_value_list[:TEST_NUMS]:
            test_img_name = cfg.IMG_DATA_PATH + i.split(' ')[0]
            test_img_index = int(i.split(' ')[1])
            target_formula = formula_list[test_img_index].split(' ')
            try:
                target_prob_index = [
                    int(properties['char_to_idx'][i]) for i in target_formula
                    if list(i)
                ]
                print('-------{}/{}------'.format(count + 1, TEST_NUMS))
                print('Input img is:', test_img_name)
                print('Input formula is:', formula_list[test_img_index])
                with open(cfg.PREDICT_PATH_LOG + 'formula_input.txt',
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
                with open(cfg.PREDICT_PATH_LOG + 'err_log.txt', 'a') as err:
                    err.write(
                        i.split(' ')[0] + '   ' + str(e) + '   ' +
                        str(formula_list[test_img_index]) + '\n')
        C_1_G = correct_1_generate / count
        C_81_G = correct_81_generate / count
        C_80_G = correct_80 / count
        C_1_NG = correct_1_Ngenerate / count
        W_NG = wrong_Ngenerate / count
        with open(cfg.PREDICT_PATH_LOG + 'result' + str(TEST_NUMS) + '.txt',
                  'w') as res:
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
    # if os.path.exists(cfg.PREDICT_PATH_LOG):
    #     shutil.rmtree(cfg.PREDICT_PATH_LOG)
    #     os.mkdir(cfg.PREDICT_PATH_LOG)
    # if not os.path.exists(cfg.PREDICT_PATH_LOG):
    #     os.mkdir(cfg.PREDICT_PATH_LOG)
    main()
