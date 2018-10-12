
'''
File: config_formula.py
Project: formula_rocog
File Created: Saturday, 28th April 2018 6:40:38 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Monday, 25th June 2018 3:23:03 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''
import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.ENV = edict()
__C.ENV.LOCAL = False  # 是否本地运行
__C.ENV.SIMPLE_ENHANCE = True  # 使用何种增强方式,是否使用简单方式进行增强

if __C.ENV.LOCAL and __C.ENV.SIMPLE_ENHANCE:
    __C.DATASET_PATH = '/Users/xiaofeng/Code/Github/dataset/formula/charactor_enhance'
if __C.ENV.LOCAL and not __C.ENV.SIMPLE_ENHANCE:
    __C.DATASET_PATH = '/Users/xiaofeng/Code/Github/dataset/formula/simple_enhance'
if not __C.ENV.LOCAL and __C.ENV.SIMPLE_ENHANCE:
    __C.DATASET_PATH = '/home/xiaofeng/data/formula/generate_enhance_no_char_formula'
if not __C.ENV.LOCAL and not __C.ENV.SIMPLE_ENHANCE:
    __C.DATASET_PATH = '/home/xiaofeng/data/formula/generate_enhance_char_formula'

__C.IMG_ORI = os.path.join(__C.DATASET_PATH, 'img_ori')
__C.DATA_ROOT = os.path.join(__C.DATASET_PATH, 'prepared')
__C.IMG_DATA_PATH = os.path.join(__C.DATA_ROOT, 'img')
__C.MODEL_SAVED = os.path.join(__C.DATASET_PATH, 'model_saved_remote')
__C.PREDICT_PATH = os.path.join(__C.DATASET_PATH, 'predict')
__C.PREDICT_PATH_LOG = os.path.join(__C.DATASET_PATH, 'predict_log')

if __C.ENV.LOCAL:
    __C.CPU_NUMS = 1
    __C.CPU_THREADS = 2
    __C.GPU = False
if not __C.ENV.LOCAL:
    __C.CPU_NUMS = 7
    __C.CPU_THREADS = 14
    __C.GPU = True
    __C.GPU_PERCENTAGE = 0.9

# if __C.ENV.LOCAL:
#     __C.DATA_ROOT = '/Users/xiaofeng/Code/Github/dataset/charactor_formula/formula/prepared'
#     __C.MODEL_SAVED = '/Users/xiaofeng/Code/Github/dataset/charactor_formula/formula/model_saved'
#     __C.PREDICT_PATH = '/Users/xiaofeng/Code/Github/dataset/charactor_formula/formula/predict/'
#     __C.PREDICT_PATH_LOG = '/Users/xiaofeng/Code/Github/dataset/charactor_formula/formula/predict_log/'
#     __C.PRETRAINED = '/Users/xiaofeng/Code/Github/dataset/pretrained_model'
#     __C.CPU_NUMS = 1
#     __C.CPU_THREADS = 2
#     __C.GPU = False

# if not __C.ENV.LOCAL:
#     __C.DATA_ROOT = '/home/xiaofeng/data/formula/generate_enhance/prepared'
#     __C.MODEL_SAVED = '/home/xiaofeng/data/formula/generate_enhance/model_saved_vgg16'
#     __C.PREDICT_PATH = '/home/xiaofeng/data/formula/generate_enhance/predict_remote/'
#     __C.PREDICT_PATH_LOG = '/home/xiaofeng/data/formula/generate_enhance/predict_remote_log/'
#     __C.PRETRAINED = '/home/xiaofeng/data/pretrained_model'
#     __C.CPU_NUMS = 7
#     __C.CPU_THREADS = 14
#     __C.GPU = True
#     __C.GPU_PERCENTAGE = 0.9


# MAKE THE DICT

__C.SIZE_LIST = [(200, 50), (120, 50), (160, 40), (360, 100), (240, 50), (280, 50), (200, 40), (280, 40),
                 (240, 40), (360, 40), (320, 40), (500, 100), (360, 60), (320, 50), (400, 50), (360, 50)]
__C.PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
__C.DATASET_LIST = ['train', 'validate']


if __C.ENV.SIMPLE_ENHANCE:
    __C.DATA_LABEL_PATH = os.path.join(__C.PROJECT_PATH, 'dataset/formula/simple_enhance/data_label')
    __C.PREPARED = os.path.join(__C.PROJECT_PATH, 'dataset/formula/simple_enhance/prepared')
else:
    __C.DATA_LABEL_PATH = os.path.join(__C.PROJECT_PATH, 'dataset/formula/charactor_enhance/data_label')
    __C.PREPARED = os.path.join(__C.PROJECT_PATH, 'dataset/formula/charactor_enhance/prepared')

__C.VOCAB_PATH = os.path.join(__C.DATA_LABEL_PATH, 'latex_vocab.txt')
__C.FORMULA_PATH = os.path.join(__C.DATA_LABEL_PATH, 'formula_normal.lst')
__C.PROPERTIES = os.path.join(__C.PREPARED, 'properties.npy')
__C.CHECKPOINT_PATH = os.path.join(__C.MODEL_SAVED, 'ckpt')
__C.SUMMARY_PATH = os.path.join(__C.MODEL_SAVED, 'log')
__C.NET_LIST = ['vgg_16', 'vgg_19', 'resnet_v2_50', 'resnet_v2_152']
__C.OPTIMIZER = ['adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'rmsprop', 'sgd']
__C.LEARNING_STYLE = ['exponential', 'fixed', 'polynomial']

__C.SAVER_SUMMARY = edict()
__C.SAVER_SUMMARY.NET_NAME_SELECT = 'vgg16'
__C.SAVER_SUMMARY.OPTIMIZER_SELECT = 'momentum'
__C.SAVER_SUMMARY.LEARNING_STYLE_SELECT = 'fixed'


def exist_or_not(pathlist):
    for path in pathlist:
        if not os.path.exists(path):
            os.makedirs(path)


exist_or_not([__C.PREDICT_PATH, __C.PREDICT_PATH_LOG, __C.CHECKPOINT_PATH, __C.SUMMARY_PATH])

# 根据词表的数量确定
# 如果使用全连接层+softmax进行损失函数的设置，使用这个参数
# VOCABLARY_SIZE = len(open(VOCAB_PATH).readlines())+3

__C.VOCABLARY_SIZE = len(open(__C.VOCAB_PATH).readlines())+3
__C.RATIO = [size[0] / size[1] for size in __C.SIZE_LIST]


# 训练参数
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.MAX_ITERS = 10000000000
__C.TRAIN.SAVED_NUMS = 500
# 学习率退化设置
__C.TRAIN.LEARNING_RATE = 0.1
__C.TRAIN.GAMMA = 0.7
__C.TRAIN.LERANING_DACEY = 1000000
__C.TRAIN.MIN_LEARNING_RATE = 0.001
__C.TRAIN.LABEL_SMOOTHING = 0.0
__C.TRAIN.DISPLAY_NUMS = 2000
__C.TRAIN.EVALUATE = 4000
__C.TRAIN.DECAY_STEPS = 200000
__C.TRAIN.DECAY_RATE = 0.5
__C.TRAIN.STAIRCASE = True
__C.TRAIN.SNAPSHOT_PREFIX = 'Ctpn_char_formulas'
__C.TRAIN.SNAPSHOT_INFIX = ''
# TEST
__C.TEST = edict()
__C.TEST.BATCH_SIZE = 1
__C.TEST.MODEL_SAVED = os.path.join(__C.PROJECT_PATH, 'checkpoint', 'formula_recog')
# network\
__C.MODEL = edict()
__C.MODEL.FEATURE = 512
__C.MODEL.DIMS_INPUT = 80
__C.MODEL.DIMS_HIDDEN = 256
__C.MODEL.DIMS_ATTENTION = 512
__C.MODEL.NUM_LAYERS = 0
__C.MODEL.DIMS_OUTPUT = __C.VOCABLARY_SIZE + 1


# def get_output_dir(imdb, weights_filename):
#     """Return the directory where experimental artifacts are placed.
#     If the directory does not exist, it is created.
#     A canonical path is built using the name from an imdb and a network
#     (if not None).
#     """
#     outdir = osp.abspath(
#         osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
#     if weights_filename is not None:
#         outdir = osp.join(outdir, weights_filename)
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#     return outdir


# def get_log_dir(imdb):
#     """Return the directory where experimental artifacts are placed.
#     If the directory does not exist, it is created.
#     A canonical path is built using the name from an imdb and a network
#     (if not None).
#     """
#     log_dir = osp.abspath(
#         osp.join(
#             __C.ROOT_DIR, 'logs', __C.LOG_DIR, imdb.name,
#             strftime("%Y-%m-%d-%H-%M-%S", localtime())))
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     return log_dir


# def _merge_a_into_b(a, b):
#     """Merge config dictionary a into config dictionary b, clobbering the
#     options in b whenever they are also specified in a.
#     """
#     if type(a) is not edict:
#         return

#     for k, v in a.items():
#         # a must specify keys that are in b
#         # if not b.has_key(k): #--python2
#         if k not in b:  # python3
#             raise KeyError('{} is not a valid config key'.format(k))

#         # the types must match, too
#         old_type = type(b[k])
#         if old_type is not type(v):
#             if isinstance(b[k], np.ndarray):
#                 v = np.array(v, dtype=b[k].dtype)
#             else:
#                 raise ValueError(('Type mismatch ({} vs. {}) '
#                                   'for config key: {}').format(
#                                       type(b[k]), type(v), k))

#         # recursively merge dicts
#         if type(v) is edict:
#             try:
#                 _merge_a_into_b(a[k], b[k])
#             except:
#                 print('Error under config key: {}'.format(k))
#                 raise
#         else:
#             b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            # assert d.has_key(subkey)
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
