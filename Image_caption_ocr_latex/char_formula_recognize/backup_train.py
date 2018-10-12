#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofeng
# Date: 2018-04-01 12:12:42
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-01 12:12:42

from PIL import Image
import tensorflow as tf
import tflib
import tflib.ops
import tflib.network
from tqdm import tqdm
import numpy as np
import data_loaders
import time
import os
import shutil
import datetime
import config as cfg
slim = tf.contrib.slim

BATCH_SIZE = 16
EMB_DIM = 80
ENC_DIM = 256
DEC_DIM = ENC_DIM * 2
NUM_FEATS_START = 64
D = NUM_FEATS_START * 8
V = cfg.V_OUT  # vocab size
NB_EPOCHS = 100000
H = 20
W = 50
PRECEPTION = 0.6
THREAD = 13
LEARNING_DECAY = 20000
IMG_PATH = cfg.IMG_DATA_PATH
PROPERTIES = cfg.PROPERTIES
ckpt_path = cfg.CHECKPOINT_PATH
summary_path = cfg.SUMMARY_PATH
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


def exist_or_not(path):
    if not os.path.exists(path):
        os.makedirs(path)


exist_or_not(summary_path)

with open('config.txt', 'w+') as f:
    cfg_dict = cfg.__dict__
    for key in sorted(cfg_dict.keys()):
        if key[0].isupper():
            cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
            f.write(cfg_str)
    f.close()
#  Create model
# X = tf.placeholder(
    # shape=(None, None, None, 3), dtype=tf.float32)  # restnet的占位符
X = tf.placeholder(shape=(None, None, None, None), dtype=tf.float32)  # 原文中的X占位符
mask = tf.placeholder(shape=(None, None), dtype=tf.int32)
seqs = tf.placeholder(shape=(None, None), dtype=tf.int32)
input_seqs = seqs[:, :-1]
target_seqs = seqs[:, 1:]
ctx = tflib.network.im2latex_cnn(X, NUM_FEATS_START, True)  # 原始repo中卷积方式
# ctx = tflib.network.vgg16(X)  # 使用vgg16的卷积方式

# 进行编码
emb_seqs = tflib.ops.Embedding('Embedding', V, EMB_DIM, input_seqs)
out, state = tflib.ops.im2latexAttention('AttLSTM', emb_seqs, ctx, EMB_DIM,
                                         ENC_DIM, DEC_DIM, D, H, W)
logits = tflib.ops.Linear('MLP.1', out, DEC_DIM, V)

# predictions = tf.argmax(tf.nn.softmax(logits[:, -1]), axis=1)
loss = tf.reshape(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(logits, [-1, V]),
        labels=tf.reshape(seqs[:, 1:], [-1])), [tf.shape(X)[0], -1])
# add paragraph ⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️
output = tf.reshape(logits, [-1, V])
output_index = tf.to_int32(tf.argmax(output, 1))
true_labels = tf.reshape(seqs[:, 1:], [-1])
correct_prediction = tf.equal(output_index, true_labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# ⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️⬆️
mask_mult = tf.to_float(mask[:, 1:])
loss = tf.reduce_sum(loss * mask_mult) / tf.reduce_sum(mask_mult)
#train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
learning_rate = tf.train.exponential_decay(
    learning_rate=cfg.LEARNING_RATE, global_step=global_step, decay_steps=cfg.DECAY_STEPS,
    decay_rate=cfg.DECAY_RATE, staircase=cfg.STAIRCASE, name='learning_rate')

#====================================进行优化器更新==================================#
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#=================================#

gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gvs]
train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)
# summary
tf.summary.scalar('larning_rate', learning_rate)
tf.summary.scalar('model_loss', loss)
# tf.summary.scalar('model_prediction', predictions)
tf.summary.scalar('model_accuracy', accuracy)
tf.summary.histogram('model_loss_his', loss)
tf.summary.histogram('model_acc_his', accuracy)
gradient_norms = [tf.norm(grad) for grad, var in gvs]
tf.summary.histogram('gradient_norm', gradient_norms)
tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
merged = tf.summary.merge_all()


# function to predict the latex
def score(set='valididate', batch_size=32):
    score_itr = data_loaders.data_iterator(set, batch_size)
    losses = []
    for score_imgs, score_seqs, score_mask in score_itr:
        _loss = sess.run(
            loss,
            feed_dict={
                X: score_imgs,
                seqs: score_seqs,
                mask: score_mask
            })
        losses.append(_loss)
    set_loss = np.mean(losses)

    perp = np.mean(list(map(lambda x: np.power(np.e, x), losses)))
    return set_loss, perp


# init = tf.global_variables_initializer()
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
config = tf.ConfigProto(intra_op_parallelism_threads=THREAD)
# config.gpu_options.per_process_gpu_memory_fraction = PRECEPTION
sess = tf.Session(config=config)
sess.run(init)
# restore the weights
saver2 = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=0.5)
saver2_path = os.path.join(ckpt_path, 'weights_best.ckpt')

file_list = os.listdir(ckpt_path)
if file_list:
    for i in file_list:
        if i == 'checkpoint':
            print('Restore the weight files form:', ckpt_path)
            saver2.restore(sess, tf.train.latest_checkpoint(ckpt_path))
suammry_writer = tf.summary.FileWriter(
    summary_path, flush_secs=60, graph=sess.graph)

coord = tf.train.Coordinator()
thread = tf.train.start_queue_runners(sess=sess, coord=coord)

losses = []
times = []
print("Compiled Train function!")
# Test is train func runs
i = 0
iter = 0
best_perp = np.finfo(np.float32).max
for i in range(i, NB_EPOCHS):
    print('best_perp', best_perp)
    costs = []
    times = []
    pred = []
    itr = data_loaders.data_iterator('train', BATCH_SIZE)
    for train_img, train_seq, train_mask in itr:
        iter += 1
        start = time.time()
        _, _loss, _acc, learn, summary = sess.run(
            [train_step, loss, accuracy, learning_rate, merged],
            feed_dict={X: train_img, seqs: train_seq, mask: train_mask
                       })
        times.append(time.time() - start)
        costs.append(_loss)
        pred.append(_acc)
        if iter % 100 == 0:
            print("Iter: %d (Epoch %d--%d)" % (iter, i + 1, NB_EPOCHS))
            print("\tMean cost: ", np.mean(costs))
            print("\tMean prediction: ", np.mean(pred))
            print("\tMean time: ", np.mean(times))
            print('\tSaveing summary to the path:', summary_path)
            print('\tSaveing model to the path:', saver2_path)
            suammry_writer.add_summary(summary, global_step=iter * i + iter)
            saver2.save(sess, saver2_path)
    print('learning rate is:', learn)
    print("\n\nEpoch %d Completed!" % (i + 1))
    print("\tMean train cost: ", np.mean(costs))
    print("\tMean train perplexity: ",
          np.mean(list(map(lambda x: np.power(np.e, x), costs))))
    print("\tMean time: ", np.mean(times))
    print('\n\n')
    print('processing the validate data...')
    val_loss, val_perp = score('validate', BATCH_SIZE)
    print("\tMean val cost: ", val_loss)
    print("\tMean val perplexity: ", val_perp)
    Info_out = datetime.datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S') + '   ' + 'iter/epoch/epoch_nums-%d/%d/%d' % (
            iter, i,
            NB_EPOCHS) + '    ' + 'val cost/val perplexity:{}/{}'.format(
                val_loss, val_perp)
    with open(summary_path + 'val_loss.txt', 'a') as file:
        file.writelines(Info_out)
    file.close()
    if val_perp < best_perp:
        best_perp = val_perp
        saver2.save(sess, saver2_path)
        print("\tBest Perplexity Till Now! Saving state!")
coord.request_stop()
coord.join(thread)
