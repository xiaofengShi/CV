
import os
import sys
from pprint import pprint

# print(sys.path)
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import parallel_reader

# import config as cfg

# 优化器
adadelta_rho = 0.95
adagrad_initial_accumulator_value = 0.1
adam_beta1 = 0.9
adam_beta2 = 0.999,
opt_epsilon = 1.0
ftrl_learning_rate_power = -0.5
ftrl_initial_accumulator_value = 0.1
ftrl_l1 = 0.0
ftrl_l2 = 0.0
momentum = 0.9
rmsprop_momentum = 0.9
rmsprop_decay = 0.9

end_learning_rate = 0.000001
# learning_rate_decay_factor = 0.7
slim = tf.contrib.slim


def configure_optimizer(optimizer_name, learning_rate):
    if optimizer_name == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=adadelta_rho, epsilon=opt_epsilon)
    elif optimizer_name == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate, initial_accumulator_value=adagrad_initial_accumulator_value)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=opt_epsilon)
    elif optimizer_name == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate, learning_rate_power=ftrl_learning_rate_power,
            initial_accumulator_value=ftrl_initial_accumulator_value, l1_regularization_strength=ftrl_l1,
            l2_regularization_strength=ftrl_l2)
    elif optimizer_name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum=momentum, name='Momentum')
    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=rmsprop_decay, momentum=rmsprop_momentum, epsilon=opt_epsilon)
    elif optimizer_name == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', optimizer_name)
    return optimizer


def configure_learning_rate(learning_rate_decay_type, learning_rate,
                            decay_steps, learning_rate_decay_rate,
                            global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """
    # decay_steps = int(
    #     num_samples_per_epoch / flags.batch_size * flags.num_epochs_per_decay)

    if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(
            learning_rate,
            global_step,
            decay_steps,
            learning_rate_decay_rate,
            staircase=True,
            name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
        # return tf.constant(learning_rate, name='fixed_learning_rate')
        return tf.Variable(learning_rate, trainable=False)
    elif learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(
            learning_rate,
            global_step,
            decay_steps,
            end_learning_rate,
            power=1.0,
            cycle=False,
            name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         learning_rate_decay_type)
