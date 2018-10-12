# top 1 accuracy 0.9249791286257038 top k accuracy 0.9747623788455786
import os
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.python.ops import control_flow_ops

from config import DIR_IMAGE_TRAIN
from preprocess import fetch_data

logger = logging.getLogger('Training a chinese print char recognition')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tf.app.flags.DEFINE_integer('charset_size', 3832,
                            "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64,
                            "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 30000, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 1000, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './models/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './data/train/',
                           'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './data/train/',
                           'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False,
                            'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_boolean('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'validation',
                           'Running mode. One of {"train", "valid", "test"}')

FLAGS = tf.app.flags.FLAGS


def build_graph(top_k):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(
        dtype=tf.float32, shape=[None, 64 * 64], name='input_image_batch')
    x_image = tf.reshape(images, [-1, 64, 64, 1], name="image_batch")
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')

    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
            normalizer_fn=slim.batch_norm,
            normalizer_params={'is_training': is_training}):
        conv3_1 = slim.conv2d(
            x_image, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
        max_pool_1 = slim.max_pool2d(
            conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
        conv3_2 = slim.conv2d(
            max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
        max_pool_2 = slim.max_pool2d(
            conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
        conv3_3 = slim.conv2d(
            max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
        max_pool_3 = slim.max_pool2d(
            conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')
        conv3_4 = slim.conv2d(
            max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
        conv3_5 = slim.conv2d(
            conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
        max_pool_4 = slim.max_pool2d(
            conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

        flatten = slim.flatten(max_pool_4)
        fc1 = slim.fully_connected(
            slim.dropout(flatten, keep_prob),
            1024,
            activation_fn=tf.nn.relu,
            scope='fc1')
        logits = slim.fully_connected(
            slim.dropout(fc1, keep_prob),
            FLAGS.charset_size,
            activation_fn=None,
            scope='fc2')
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)

    global_step = tf.get_variable(
        "step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = slim.learning.create_train_op(
        loss, optimizer, global_step=global_step)
    probabilities = tf.nn.softmax(logits)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(
        probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(
        tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {
        'images': images,
        'labels': labels,
        'keep_prob': keep_prob,
        'top_k': top_k,
        'global_step': global_step,
        'train_op': train_op,
        'loss': loss,
        'is_training': is_training,
        'accuracy': accuracy,
        'accuracy_top_k': accuracy_in_top_k,
        'merged_summary_op': merged_summary_op,
        'predicted_distribution': probabilities,
        'predicted_index_top_k': predicted_index_top_k,
        'predicted_val_top_k': predicted_val_top_k
    }


def train():
    print('Begin training')
    # ======================================================
    # train_feeder = DataIterator(data_dir='../data/train/')
    # test_feeder = DataIterator(data_dir='../data/test/')
    # ======================================================

    model_name = 'chinese-rec-model'
    with tf.Session() as sess:

        # ==================================================
        # train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        # test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        # ==================================================

        graph = build_graph(top_k=1)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')
        try:
            i = 0
            while not coord.should_stop():
                i += 1
                start_time = time.time()

                # train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                train_images_batch, train_labels_batch = fetch_data(
                    128, DIR_IMAGE_TRAIN)
                feed_dict = {
                    graph['images']: train_images_batch,
                    graph['labels']: train_labels_batch,
                    graph['keep_prob']: 0.8,
                    graph['is_training']: True
                }
                _, loss_val, train_summary, step = sess.run(
                    [
                        graph['train_op'], graph['loss'],
                        graph['merged_summary_op'], graph['global_step']
                    ],
                    feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                end_time = time.time()
                logger.info("the step {0} takes {1} loss {2}".format(
                    step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    break
                if step % FLAGS.eval_steps == 1:
                    test_images_batch, test_labels_batch = fetch_data(
                        128, DIR_IMAGE_TRAIN)
                    feed_dict = {
                        graph['images']: test_images_batch,
                        graph['labels']: test_labels_batch,
                        graph['keep_prob']: 1.0,
                        graph['is_training']: False
                    }
                    accuracy_test, test_summary = sess.run(
                        [graph['accuracy'], graph['merged_summary_op']],
                        feed_dict=feed_dict)
                    if step > 300:
                        test_writer.add_summary(test_summary, step)
                    logger.info(
                        '===============Eval a batch=======================')
                    logger.info('the step {0} test accuracy: {1}'.format(
                        step, accuracy_test))
                    logger.info(
                        '===============Eval a batch=======================')
                if step % FLAGS.save_steps == 1:
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(
                        sess,
                        os.path.join(FLAGS.checkpoint_dir, model_name),
                        global_step=graph['global_step'])
        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
            saver.save(
                sess,
                os.path.join(FLAGS.checkpoint_dir, model_name),
                global_step=graph['global_step'])
        finally:
            coord.request_stop()
        coord.join(threads)


def validation():
    print('Begin validation')

    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session() as sess:

        graph = build_graph(top_k=3)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()
                 )  # initialize test_feeder's inside state

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        logger.info(':::Start validation:::')
        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = fetch_data(
                    128, DIR_IMAGE_TRAIN)
                feed_dict = {
                    graph['images']: test_images_batch,
                    graph['labels']: test_labels_batch,
                    graph['keep_prob']: 1.0,
                    graph['is_training']: False
                }
                batch_labels, probs, indices, acc_1, acc_k = sess.run(
                    [
                        graph['labels'], graph['predicted_val_top_k'],
                        graph['predicted_index_top_k'], graph['accuracy'],
                        graph['accuracy_top_k']
                    ],
                    feed_dict=feed_dict)
                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()
                logger.info(
                    "the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
                    .format(i, end_time - start_time, acc_1, acc_k))

        except tf.errors.OutOfRangeError:
            logger.info(
                '==================Validation Finished================')


#
#            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
#            acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
#            logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
        finally:
            coord.request_stop()
        coord.join(threads)
    return {
        'prob': final_predict_val,
        'indices': final_predict_index,
        'groundtruth': groundtruth
    }


def inference(image):
    print('inference')
    from preprocess import get_X
    temp_image = get_X(image)
    with tf.Session() as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        predict_val, predict_index = sess.run(
            [graph['predicted_val_top_k'], graph['predicted_index_top_k']],
            feed_dict={
                graph['images']: temp_image,
                graph['keep_prob']: 1.0,
                graph['is_training']: False
            })
    return predict_val, predict_index


def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'validation':
        dct = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(dct, f)
        logger.info('Write file ends')
    elif FLAGS.mode == 'inference':
        image_path = 'E:/captcha-data/ocr/bold/中 (6).jpg'
        final_predict_val, final_predict_index = inference(image_path)
        logger.info(
            'the result info label {0} predict index {1} predict_val {2}'.
            format(190, final_predict_index, final_predict_val))


if __name__ == "__main__":
    tf.app.run()