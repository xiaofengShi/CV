# -*- coding: utf-8 -*- 
# _Author_: xiaofengShi 
# Date: 2018-03-18 18:59:05 
# Last Modified by:   xiaofengShi 
# Last Modified time: 2018-03-18 18:59:05 
 
from net import net_tiny
import tensorflow as tf
import config as cfg
import os
import time
import datetime
from dataset.data_to_tfrecord import run_dataset_tfrecord
from util.timer import Timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

slim = tf.contrib.slim

# 保存cfg文件
with open(os.path.join(cfg.TRAINING_PROCESS_CONFIG_DIR, 'config.txt'), 'w+') as f:
    cfg_dict = cfg.__dict__
    for key in sorted(cfg_dict.keys()):
        if key[0].isupper():
            cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
            f.write(cfg_str)
    f.close()


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


image_holder = tf.placeholder(tf.float32,
                              [cfg.BATCH_SIZE, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.IMAGE_CHANNEL])
label_holder = tf.placeholder(tf.int32, [cfg.BATCH_SIZE, cfg.CLASSES])

# image_batch, label_batch = run_dataset_tfrecord()
with slim.arg_scope(net_tiny.tiny_net_arg_scope()):
    pred, end_point = net_tiny.tiny_net(inputs=image_holder, is_training=True)


def loss2(logits, labels):
    flogits = []
    fgclasses = []
    for i in range(len(logits)):
        flogits.append(tf.reshape(logits[i], [-1, cfg.CLASSES]))
        fgclasses.append(tf.reshape(labels[i], [-1]))
    pred = tf.concat(flogits, axis=0)
    truth = tf.concat(fgclasses, axis=0)
    top_in_k = tf.nn.top_k(pred, truth, 1)
    loss_2 = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(pred, truth))
    tf.add_to_collection('loss', loss_2)
    tf.summary.scalar('loss', loss_2)
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), top_in_k, loss_2


global_step = tf.get_variable(
    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
# global_step = tf.train.global_step()
learning_rate = tf.train.exponential_decay(learning_rate=cfg.LEARNING_RATE, global_step=global_step,
                                           decay_steps=cfg.DECAY_STEPS, decay_rate=cfg.DECAY_RATE,
                                           staircase=cfg.STAIRCASE, name='learning_rate')
tf.summary.scalar('learning_rate', learning_rate)
cross_loss = loss(pred, label_holder)
tf.summary.scalar('loss', cross_loss)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_loss, global_step)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(label_holder, 1))
# accuracy = tf.nn.in_top_k(pred, label_batch, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# top_k_op = tf.nn.in_top_k(predictions=pred, targets=label_holder, k=1)
tf.summary.scalar('prediction', accuracy)
merged = tf.summary.merge_all()
initop = tf.group(tf.global_variables_initializer(),
                  tf.local_variables_initializer())
ckpt_file = os.path.join(cfg.MODEL_OUTPUT_DIR, 'save.ckpt')

config = tf.ConfigProto(device_count={"CPU": 1},  # limit to num_cpu_core CPU usage
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=2,  # limit the threads
                        log_device_placement=False)
image_batch, label_batch = run_dataset_tfrecord(
    is_training=True, shuffling=False)
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=0.5)
with tf.Session(config=config) as sess:
    print('Start training...')
    sess.run(initop)
    txt = open(os.path.join(cfg.TRAINING_PROCESS_CONFIG_DIR,
                            'traing_process.txt'), 'w+')
    train = Timer()
    if cfg.WEIGHT_DIR is not None and cfg.META_DIR is not None:
        print('Restoring weights from: ' + cfg.WEIGHT_DIR)
        # saver = tf.train.import_meta_graph(cfg.META_DIR)
        saver.restore(sess, tf.train.latest_checkpoint(cfg.WEIGHT_DIR))
    writer = tf.summary.FileWriter(cfg.SUMMARY_SAVED, flush_secs=60)
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch in range(cfg.EPOCH_NUMS):
        for inter in range(1, cfg.ITER + 1):
            start_time = time.time()
            image_feed_batch, label_feed_batch = sess.run(
                [image_batch, label_batch])
            _, cost, acc, learning, summary = \
                sess.run([optimizer, cross_loss, accuracy, learning_rate, merged],
                         feed_dict={image_holder: image_feed_batch,
                                    label_holder: label_feed_batch})
            duration = time.time() - start_time
            logits = sess.run(pred, feed_dict={
                              image_holder: image_feed_batch, label_holder: label_feed_batch})
            # softmax_pred = sess.run(tf.nn.softmax(logits=logits))
            prediction_label = sess.run(tf.argmax(tf.nn.softmax(logits), 1))
            input_label = sess.run(tf.argmax(label_feed_batch, 1))

            # savepath = saver.save(sess, ckpt_file)
            # print('saved')
            if (inter + (epoch) * cfg.ITER) % cfg.DISPLAY_STEP == 0:
                log = '%s-%s/%s-%s/%s :loss is : %.5f, accuracy is : %5f,learning rate is :% s' % (
                    datetime.datetime.now().strftime(
                        '%m/%d %H:%M:%S'), inter, cfg.ITER, epoch, cfg.EPOCH_NUMS, cost,
                    acc, learning)
                print(log)
                print('predict', prediction_label)
                print('input_label', input_label)
                writer.add_summary(summary, inter * epoch * cfg.BATCH_SIZE)
            if (inter + epoch * cfg.ITER) % 50 == 0:
                # writer.add_summary(summary, inter * epoch * cfg.BATCH_SIZE)
                savepath = saver.save(sess, ckpt_file)
                examples_per_sec = duration / cfg.BATCH_SIZE
                sec_per_batch = float(duration)
                print('example per sec is :', examples_per_sec,
                      'second per batch is :', sec_per_batch)
                print('Summary and ckpt are saving to ', savepath)
                remain_time = train.remain(
                    inter + epoch * cfg.ITER, cfg.EPOCH_NUMS * cfg.ITER)
                print('Current step is %s , Total step is %s ,Remaining time is : %s'
                      % (inter + epoch * cfg.ITER, cfg.EPOCH_NUMS * cfg.ITER, remain_time))
                txt.write(log)

    coord.request_stop()
    coord.join(thread)
    print("training finish!")
    txt.close()
