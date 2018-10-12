# 判断是否是在本地电脑运行
import os

A_NOTE = '路径存储信息'
LOCAL = False  # 是否本地运行
ENHANCE = False
if not LOCAL and not ENHANCE:
    DATA_ROOT = '/home/xiaofeng/data/char_formula/prepared/'
    MODEL_SAVED = '/home/xiaofeng/data/char_formula/prepared/model_saved'


# 进行资源设置
if not LOCAL:
    CPU_NUMS = 7
    CPU_THREADS = 14
    GPU = True
    GPU_PERCENTAGE = 0.9
    PRETRAINED = '/home/xiaofeng/data/pretrained_model/'
if LOCAL:
    CPU_NUMS = 1
    CPU_THREADS = 2
    GPU = False
    PRETRAINED = '/Users/xiaofeng/Code/Github/dataset/pretrained_model/'

if not os.path.exists(MODEL_SAVED):
    os.makedirs(MODEL_SAVED)
SIZE_LIST = [(120, 50), (160, 40), (240, 40), (320, 50), (360, 60), (200, 40),
             (200, 50), (500, 100), (240, 50), (320, 40), (280, 50), (280, 40),
             (360, 100), (400, 50), (360, 50), (360, 40)]
SET_LIST = ['train', 'validate']
VOCAB_PATH = DATA_ROOT + 'char_formula_full_db.txt'
FORMULA_PATH = DATA_ROOT + 'new_char_formula_normal.ls'
IMG_DATA_PATH = DATA_ROOT + 'img/'

CHECKPOINT_PATH = os.path.join(MODEL_SAVED, 'ckpt')
SUMMARY_PATH = os.path.join(MODEL_SAVED, 'log')
PROPERTIES = DATA_ROOT + 'properties.npy'
NET_LIST = ['vgg_16', 'vgg_19', 'resnet_v2_50', 'resnet_v2_152']
OPTIMIZER = ['adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'rmsprop', 'sgd']
LEARNING_STYLE = ['exponential', 'fixed', 'polynomial']


def exist_or_not(pathlist):
    for path in pathlist:
        if not os.path.exists(path):
            os.makedirs(path)


exist_or_not([SUMMARY_PATH, CHECKPOINT_PATH])

# 根据词表的数量确定
V_OUT = 1 + len(open(VOCAB_PATH).readlines())
# V_OUT = 3 + len(open(VOCAB_PATH).readlines())
RATIO = [size[0] / size[1] for size in SIZE_LIST]

# 训练参数
BATCH_SIZE = 32
EPOCH_NUMS = 10000
SAVE_ITER = 100
SUMMARY_ITER = 200
# 学习率退化设置
LEARNING_RATE = 0.1
MIN_LEARNING_RATE = 0.001
LABEL_SMOOTHING = 0.0

DISPLAY_NUMS = 100
SAVED_NUMS = 500
DECAY_STEPS = 10000
DECAY_RATE = 0.7
STAIRCASE = True
