import tensorflow as tf
import time
import numpy as np
import theano
theano.config.floatX = 'float32'
import lasagne
import tflib

from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn


def initializer(name, shape, val=0, gain='linear', std=0.01, mean=0.0, range=0.01, alpha=0.01):
    """
    Wrapper function to perform weight initialization using standard techniques

    :parameters:
        name: Name of initialization technique. Follows same names as lasagne.init module
        shape: list or tuple containing shape of weights
        val: Fill value in case of constant initialization
        gain: one of 'linear','sigmoid','tanh', 'relu' or 'leakyrelu'
        std: standard deviation used for normal / uniform initialization
        mean: mean value used for normal / uniform initialization
        alpha: used when gain = 'leakyrelu'
    """

    if gain in ['linear', 'sigmoid', 'tanh']:
        gain = 1.0
    elif gain == 'leakyrelu':
        gain = np.sqrt(2 / (1 + alpha**2))
    elif gain == 'relu':
        gain = np.sqrt(2)
    else:
        raise NotImplementedError

    if name == 'Constant':
        return lasagne.init.Constant(val).sample(shape)
    elif name == 'Normal':
        return lasagne.init.Normal(std, mean).sample(shape)
    elif name == 'Uniform':
        return lasagne.init.Uniform(
            range=range, std=std, mean=mean).sample(shape)
    elif name == 'GlorotNormal':
        return lasagne.init.GlorotNormal(gain=gain).sample(shape)
    elif name == 'GlorotUniform':
        return lasagne.init.GlorotUniform(gain=gain).sample(shape)
    elif name == 'HeNormal':
        return lasagne.init.HeNormal(gain=gain).sample(shape)
    elif name == 'HeUniform':
        return lasagne.init.HeUniform(gain=gain).sample(shape)
    elif name == 'Orthogonal':
        return lasagne.init.Orthogonal(gain=gain).sample(shape)
    else:
        return lasagne.init.GlorotUniform(gain=gain).sample(shape)


def Embedding(name, n_symbols, output_dim, indices):
    """
    Creates an embedding matrix of dimensions n_symbols x output_dim upon first use.
    Looks up embedding vector for each input symbol

    :parameters:
        name: name of embedding matrix tensor variable
        n_symbols: No. of input symbols
        output_dim: Embedding dimension
        indices: input symbols tensor
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
        embedding_map = tf.get_variable(
            name=name,
            shape=[n_symbols, output_dim],
            initializer=tf.random_uniform_initializer(minval=-0.08,
                                                      maxval=0.08))
        seq_embeddings = tf.nn.embedding_lookup(embedding_map, indices)

        return seq_embeddings


def Linear(name, inputs, input_dim, output_dim, activation='linear', bias=True, init=None, weightnorm=False, **kwargs):
    """
    Compute a linear transform of one or more inputs, optionally with a bias.
    Supports more than 2 dimensions. (in which case last axis is considered the dimension to be transformed)

    :parameters:
        input_dim: tuple of ints, or int; dimensionality of the input
        output_dim: int; dimensionality of output
        activation: 'linear','sigmoid', etc. ; used as gain parameter for weight initialization ;
                     DOES NOT APPLY THE ACTIVATION MENTIONED IN THIS PARAMETER
        bias: flag that denotes whether bias should be applied
        init: name of weight initializer to be used
        weightnorm: flag that denotes whether weight normalization should be applied
    """

    with tf.name_scope(name) as scope:
        weight_values = initializer(
            init, (input_dim, output_dim), gain=activation, **kwargs)

        weight = tflib.param(name + '.W', weight_values)

        batch_size = None

        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(weight_values), axis=0))
            # nort.m_values = np.linalg.norm(weight_values, axis=0)

            target_norms = tflib.param(name + '.g', norm_values)

            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(
                    tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / norms)

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(
                result,
                tf.stack(tf.unstack(tf.shape(inputs))[:-1] + [output_dim]))

        if bias:
            b = tflib.param(name + '.b',
                            np.zeros((output_dim, ), dtype='float32'))

            result = tf.nn.bias_add(result, b)

        return result


def conv2d(name, input, kernel, stride, depth, num_filters, init='GlorotUniform', pad='SAME', bias=True,
           weightnorm=False, batchnorm=False, is_training=True, **kwargs):
    """
    Performs 2D convolution on input in NCHW data format

    :parameters:
        input - input to be convolved
        kernel - int; size of convolutional kernel
        stride - int; horizontal / vertical stride to be used
        depth - int; no. of channels of input
        num_filters - int; no. of output channels required
        batchnorm - flag that denotes whether batch normalization should be applied
        is_training - flag that denotes batch normalization mode
    """
    with tf.name_scope(name) as scope:
        filter_values = initializer(
            init, (kernel, kernel, depth, num_filters), gain='relu', **kwargs)
        filters = tflib.param(name + '.W', filter_values)

        if weightnorm:
            norm_values = np.sqrt(
                np.sum(np.square(filter_values), axis=(0, 1, 2)))
            target_norms = tflib.param(name + '.g', norm_values)
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(
                    tf.reduce_sum(
                        tf.square(filters), reduction_indices=[0, 1, 2]))
                filters = filters * (target_norms / norms)

        out = tf.nn.conv2d(input, filters, strides=[1, stride, stride, 1], padding=pad)

        if bias:
            b = tflib.param(name + '.b', np.zeros(
                num_filters, dtype=np.float32))

            out = tf.nn.bias_add(out, b)

        if batchnorm:
            out = tf.contrib.layers.batch_norm(
                inputs=out, scope=scope, is_training=is_training, data_format='NHWC')

        return out


def max_pool(name, l_input, k, s):
    """
    Max pooling operation with kernel size k and stride s on input with NCHW data format

    :parameters:
        l_input: input in NCHW data format
        k: tuple of int, or int ; kernel size
        s: tuple of int, or int ; stride value
    """

    if type(k) == int:
        k1 = k
        k2 = k
    else:
        k1 = k[0]
        k2 = k[1]
    if type(s) == int:
        s1 = s
        s2 = s
    else:
        s1 = s[0]
        s2 = s[1]
    return tf.nn.max_pool(
        l_input,
        ksize=[1, k1, k2, 1],
        strides=[1, s1, s2, 1],
        padding='SAME',
        name=name,
        data_format='NHWC')


def norm(name, l_input, lsize=4):
    """
    Wrapper function to perform local response normalization (ref. Alexnet)
    """
    return tf.nn.lrn(
        l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


class GRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, inputs, state, scope=None):
        gates = tf.nn.sigmoid(
            tflib.ops.Linear(self._name + '.Gates',
                             tf.concat(axis=1, values=[inputs, state]),
                             self._n_in + self._n_hid, 2 * self._n_hid))

        update, reset = tf.split(axis=1, num_or_size_splits=2, value=gates)
        scaled_state = reset * state

        candidate = tf.tanh(
            tflib.ops.Linear(self._name + '.Candidate',
                             tf.concat(axis=1, values=[inputs, scaled_state]),
                             self._n_in + self._n_hid, self._n_hid))

        output = (update * candidate) + ((1 - update) * state)

        return output, output


def GRU(name, inputs, n_in, n_hid):
    """
    Compute recurrent memory states using Gated Recurrent Units

    :parameters:
        n_in : int ; Dimensionality of input
        n_hid : int ; Dimensionality of hidden state / memory state
    """
    h0 = tflib.param(name + '.h0', np.zeros(n_hid, dtype='float32'))
    batch_size = tf.shape(inputs)[0]
    h0 = tf.reshape(
        tf.tile(h0, tf.stack([batch_size])), tf.stack([batch_size, n_hid]))
    return tf.nn.dynamic_rnn(
        GRUCell(name, n_in, n_hid), inputs, initial_state=h0,
        swap_memory=True)[0]


class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid, forget_bias=1.0):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, inputs, state, scope=None):
        c_tm1, h_tm1 = tf.split(axis=1, num_or_size_splits=2, value=state)
        gates = tflib.ops.Linear(
            self._name + '.Gates',
            tf.concat(axis=1, values=[inputs, h_tm1]),
            self._n_in + self._n_hid,
            4 * self._n_hid,
            activation='sigmoid')

        i_t, f_t, o_t, g_t = tf.split(
            axis=1, num_or_size_splits=4, value=gates)

        c_t = tf.nn.sigmoid(f_t + self._forget_bias) * c_tm1 + tf.nn.sigmoid(
            i_t) * tf.tanh(g_t)
        h_t = tf.nn.sigmoid(o_t) * tf.tanh(c_t)

        new_state = tf.concat(axis=1, values=[c_t, h_t])

        return h_t, new_state


def LSTM(name, inputs, n_in, n_hid, h0):
    """
    Compute recurrent memory states using Long Short-Term Memory units

    :parameters:
        n_in : int ; Dimensionality of input
        n_hid : int ; Dimensionality of hidden state / memory state
    """
    batch_size = tf.shape(inputs)[0]
    if h0 is None:
        h0 = tflib.param(name + '.init.h0', np.zeros(
            2 * n_hid, dtype='float32'))
        h0 = tf.reshape(
            tf.tile(h0_1, tf.stack([batch_size])),
            tf.stack([batch_size, 2 * n_hid]))

    return tf.nn.dynamic_rnn(
        LSTMCell(name, n_in, n_hid),
        inputs,
        initial_state=h0,
        swap_memory=True)


def BiLSTM(name, inputs, n_in, n_hid, h0_1=None, h0_2=None):
    """
    Compute recurrent memory states using Bidirectional Long Short-Term Memory units

    :parameters:
        n_in : int ; Dimensionality of input
        n_hid : int ; Dimensionality of hidden state / memory state
        h0_1: vector ; Initial hidden state of forward LSTM
        h0_2: vector ; Initial hidden state of backward LSTM
    """

    batch_size = tf.shape(inputs)[0]
    if h0_1 is None:
        h0_1 = tflib.param(name + '.init.h0_1',
                           np.zeros(2 * n_hid, dtype='float32'))
        h0_1 = tf.reshape(
            tf.tile(h0_1, tf.stack([batch_size])),
            tf.stack([batch_size, 2 * n_hid]))

    if h0_2 is None:
        h0_2 = tflib.param(name + '.init.h0_2',
                           np.zeros(2 * n_hid, dtype='float32'))
        h0_2 = tf.reshape(
            tf.tile(h0_2, tf.stack([batch_size])),
            tf.stack([batch_size, 2 * n_hid]))

    cell_fw = LSTMCell(name + '_fw', n_in, n_hid)
    cell_bw = LSTMCell(name + '_bw', n_in, n_hid)

    # with tf.variable_scope(name + '_fw_single'):
    #     cell_fw = tf.contrib.rnn.LSTMCell(n_hid)
    # with tf.variable_scope(name + '_bw_single'):
    #     cell_bw = tf.contrib.rnn.LSTMCell(n_hid)

    # '''
    # 添加多层
    # '''
    # stack_fw, stack_bw=[], []
    # for i in range(3):
    #     with tf.variable_scope(name+'_fw_{}'.format(i)):
    #         stack_fw.append(cell1)
    #     with tf.variable_scope(name+'bw_{}'.format(i)):
    #         stack_bw.append(cell2)
    # with tf.variable_scope(name + '_mul_fw'):
    #      mcell_fw = tf.contrib.rnn.MultiRNNCell(stack_fw)
    # with tf.variable_scope(name + '_mul_bw'):
    #      mcell_bw = tf.contrib.rnn.MultiRNNCell(stack_bw)

    # print(np.shape(mcell_bw))
    seq_len = tf.tile(tf.expand_dims(tf.shape(inputs)[1], 0), [batch_size])
    outputs = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        inputs,
        sequence_length=seq_len,
        initial_state_fw=h0_1,
        initial_state_bw=h0_2,
        swap_memory=True)
    return tf.concat(axis=2, values=[outputs[0][0], outputs[0][1]])


'''
Attentional Decoder as proposed in HarvardNLp paper (https://arxiv.org/pdf/1609.04938v1.pdf)
'''
ctx_vector = []


class im2latexAttentionCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, name, n_in, n_hid, L, D, ctx, forget_bias=1.0):
        self._n_in = n_in
        self._n_hid = n_hid
        self._name = name
        self._forget_bias = forget_bias
        self._ctx = ctx
        self._L = L
        self._D = D

    @property
    def state_size(self):
        return self._n_hid

    @property
    def output_size(self):
        return self._n_hid

    def __call__(self, _input, state, scope=None):

        h_tm1, c_tm1, output_tm1 = tf.split(axis=1, num_or_size_splits=3, value=state)

        gates = tflib.ops.Linear(
            self._name + '.Gates',
            tf.concat(axis=1, values=[_input, output_tm1]),
            self._n_in + self._n_hid,
            4 * self._n_hid,
            activation='sigmoid')

        i_t, f_t, o_t, g_t = tf.split(
            axis=1, num_or_size_splits=4, value=gates)

        # removing forget_bias
        c_t = tf.nn.sigmoid(f_t) * c_tm1 + tf.nn.sigmoid(i_t) * tf.tanh(g_t)
        h_t = tf.nn.sigmoid(o_t) * tf.tanh(c_t)

        target_t = tf.expand_dims(
            tflib.ops.Linear(self._name + '.target_t', h_t, self._n_hid, self._n_hid, bias=False), 2)
        # target_t = tf.expand_dims(h_t,2) # (B, HID, 1)
        a_t = tf.nn.softmax(tf.matmul(self._ctx, target_t)[:, :, 0], name='a_t')  # (B, H*W, D) * (B, D, 1)

        def _debug_bkpt(val):
            global ctx_vector
            ctx_vector = []
            ctx_vector += [val]
            return False

        debug_print_op = tf.py_func(_debug_bkpt, [a_t], [tf.bool])
        with tf.control_dependencies(debug_print_op):
            a_t = tf.identity(a_t, name='a_t_debug')

        a_t = tf.expand_dims(a_t, 1)  # (B, 1, H*W)
        z_t = tf.matmul(a_t, self._ctx)[:, 0]
        # a_t = tf.expand_dims(a_t,2)
        # z_t = tf.reduce_sum(a_t*self._ctx,1)

        output_t = tf.tanh(
            tflib.ops.Linear(
                self._name + '.output_t', tf.concat(axis=1, values=[h_t, z_t]),
                self._D + self._n_hid, self._n_hid, bias=False, activation='tanh'))

        new_state = tf.concat(axis=1, values=[h_t, c_t, output_t])

        return output_t, new_state


def im2latexAttention(name, inputs, ctx, input_dim, ENC_DIM, DEC_DIM, D, H, W):
    """
    Function that encodes the feature grid extracted from CNN using BiLSTM encoder
    and decodes target sequences using an attentional decoder mechanism

    PS: Feature grid can be of variable size (as long as size is within 'H' and 'W')

    :parameters:
        ctx - (N,C,H,W) format ; feature grid extracted from CNN
        input_dim - int ; Dimensionality of input sequences (Usually, Embedding Dimension)
        ENC_DIM - int; Dimensionality of BiLSTM Encoder
        DEC_DIM - int; Dimensionality of Attentional Decoder
        D - int; No. of channels in feature grid
        H - int; Maximum height of feature grid
        W - int; Maximum width of feature grid
    """
    # 已经对dataloader进行修改，此处的输出为NHWC
    # V = tf.transpose(ctx, [0, 2, 3, 1])  # NCHW --->>>>(N, H, W, C)
    # V = ctx
    V_cap = []
    batch_size = tf.shape(ctx)[0]
    # 创建隐藏层节点
    h0_i_1 = tf.tile(
        tflib.param(name + '.Enc_.init.h0_1',
                    np.zeros((1, H, 2 * ENC_DIM)).astype('float32')),
        [batch_size, 1, 1])

    h0_i_2 = tf.tile(tflib.param(name + '.Enc_init.h0_2',
                                 np.zeros((1, H, 2 * ENC_DIM)).astype('float32')),
                     [batch_size, 1, 1])

    def fn(prev_out, i):
        return tflib.ops.BiLSTM(
            name=name + '.BiLSTMEncoder', inputs=ctx[:, i],
            n_in=D, n_hid=ENC_DIM, h0_1=h0_i_1[:, i],
            h0_2=h0_i_2[:, i])

    #-=======================================================================#
    # 在高度的维度上运行fn方程,动态单层双向RNN
    V_cap = tf.scan(fn, tf.range(tf.shape(ctx)[1]), initializer=tf.placeholder(
        shape=(None, None, 2 * ENC_DIM), dtype=tf.float32))
    V_t = tf.reshape(
        tf.transpose(V_cap, [1, 0, 2, 3]),
        [tf.shape(inputs)[0], -1, ENC_DIM * 2])  # (B, L, ENC_DIM)

    h0_dec = tf.tile(
        tflib.param(name + '.Decoder.init.h0',
                    np.zeros((1, 3 * DEC_DIM)).astype('float32')),
        [batch_size, 1])

    cell = tflib.ops.im2latexAttentionCell(
        name=name + '.AttentionCell', n_in=input_dim, n_hid=DEC_DIM, L=H * W, D=2 * ENC_DIM, ctx=V_t)
    seq_len = tf.tile(tf.expand_dims(tf.shape(inputs)[1], 0), [batch_size])

    out = tf.nn.dynamic_rnn(cell, inputs, initial_state=h0_dec, sequence_length=seq_len, swap_memory=True)
    return out


def ownAttention(name, inputs, ctx, input_dim, ENC_DIM, DEC_DIM, D, H, W):
    """
    Function that encodes the feature grid extracted from CNN using BiLSTM encoder
    and decodes target sequences using an attentional decoder mechanism

    PS: Feature grid can be of variable size (as long as size is within 'H' and 'W')

    :parameters:
        ctx - (N,C,H,W) format ; feature grid extracted from CNN
        input_dim - int ; Dimensionality of input sequences (Usually, Embedding Dimension)
        ENC_DIM - int; Dimensionality of BiLSTM Encoder
        DEC_DIM - int; Dimensionality of Attentional Decoder
        D - int; No. of channels in feature grid
        H - int; Maximum height of feature grid
        W - int; Maximum width of feature grid
    """
    V_cap = []
    batch_size = tf.shape(ctx)[0]
    num_layers = 5

    def fn(pre):
        with tf.variable_scope('encoder_rnn', initializer=tf.orthogonal_initializer()):
            cell_fw = tf.contrib.rnn.MultiRNNCell(
                cells=[tf.nn.rnn_cell.LSTMCell(ENC_DIM)
                       for _ in range(num_layers)])
            cell_bw = tf.contrib.rnn.MultiRNNCell(
                cells=[tf.nn.rnn_cell.LSTMCell(ENC_DIM)
                       for _ in range(num_layers)])
            outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_bw=cell_bw,
                                                                   cell_fw=cell_fw,
                                                                   inputs=pre,
                                                                   sequence_length=tf.fill([batch_size],
                                                                                           tf.shape(pre)[1]),
                                                                   dtype='float32',
                                                                   swap_memory=True)
        return tf.concat(values=outputs, axis=2)

    fun = tf.make_template('fun', fn)
    rows_first = tf.transpose(ctx, [1, 0, 2, 3])
    res = tf.map_fn(fun, rows_first, dtype=tf.float32)
    encoder_output = tf.transpose(res, [1, 0, 2, 3])  # batch*length*2encode_dim
    encoder_output = tf.reshape(encoder_output, [batch_size, -1, 2 * ENC_DIM])

    cell = tflib.ops.im2latexAttentionCell(
        name=name + '.AttentionCell', n_in=input_dim, n_hid=DEC_DIM, L=H*W, D=2*ENC_DIM, ctx=encoder_output)

    h0_dec = tf.tile(
        tf.get_variable(
            name + '.Decoder.init.h0', (1, 3 * DEC_DIM),
            tf.float32, initializer=tf.orthogonal_initializer()),
        [batch_size, 1])
    seq_len = tf.tile(tf.expand_dims(tf.shape(inputs)[1], 0), [batch_size])

    out = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, initial_state=h0_dec,
                            sequence_length=seq_len, swap_memory=True)
    return out
