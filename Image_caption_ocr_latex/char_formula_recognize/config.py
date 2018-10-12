# 判断是否是在本地电脑运行
import os

A_NOTE = '路径存储信息'
LOCAL = False  # 是否本地运行
GENERATE = True  # 是否只是用数据库生成的数据进行训练
ENHANCE = True
# ==========================本地不使用数据库=========================#
if LOCAL and not GENERATE and not ENHANCE:
    DATA_ROOT = '/Users/xiaofeng/Code/Github/dataset/formula/original_data/prepared/'
    MODEL_SAVED = '/Users/xiaofeng/Code/Github/dataset/formula/original_data/model_saved/'
    PREDICT_PATH = '/Users/xiaofeng/Code/Github/dataset/formula/original_data/predict/'

# ===========================远程不使用数据库=========================#
elif not LOCAL and not GENERATE and not ENHANCE:
    DATA_ROOT = '/home/xiaofeng/data/formula/prepared/'
    MODEL_SAVED = '/home/xiaofeng/data/formula/model_saved_remote/'
    PREDICT_PATH = '/home/xiaofeng/data/formula/predict_remote/'

# =============================本地生成不增强=========================#
if LOCAL and GENERATE and not ENHANCE:
    DATA_ROOT = '/Users/xiaofeng/Code/Github/dataset/charactor_formula/formula/prepared/'
    MODEL_SAVED = '/Users/xiaofeng/Code/Github/dataset/charactor_formula/formula/prepared/'
    PREDICT_PATH = '/Users/xiaofeng/Code/Github/dataset/charactor_formula/original_data/predict/'
    PREDICT_PATH_LOG = '/Users/xiaofeng/Desktop/predict/log/'

# ===========================本地使用数据库增强==========================#
elif LOCAL and GENERATE and ENHANCE:
    # 模型训练参数
    DATA_ROOT = '/Users/xiaofeng/Code/Github/dataset/charactor_formula/formula/prepared'
    MODEL_SAVED = '/Users/xiaofeng/Code/Github/dataset/formula/enhance/model_saved/'
    PREDICT_PATH = '/Users/xiaofeng/Code/Github/dataset/formula/enhance/predict/'
    # 数据增强参数
    FORMULA_TXT = '/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/formula.txt'
    NAME_TXT = '/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/name.txt'
    DATASET_FILE = "/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/im2latex.lst"
    NEW_FORMULA_FILE = "/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/im2latex_formulas.lst"
    TRAIN_LIST = '/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/train.list'
    VALIDATE_LIST = '/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/validate.list'
    IMAGE_DIR = '/Users/xiaofeng/Code/Github/dataset/formula/enhance/img_ori'

    DATASET_FILE_ENHANCE = "/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/im2latex_enhance.lst"
    NEW_FORMULA_FILE_ENHANCE = "/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/im2latex_formulas_enhance.lst"
    NORM_FORMULA_FILE_ORI = '/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/formulas.norm_ori.lst'
    NORM_FORMULA_FILE_ENHANCE = '/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/formulas.norm_enhance.lst'
    TRAIN_LIST_ENHANCE = '/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/train_enhance.list'
    VALIDATE_LIST_ENHANCE = '/Users/xiaofeng/Code/Github/graphic/FORMULA_OCR_UNIFORM/dataset/generate/validate_enhance.list'
    IMAGE_DIR_ENHANCE = '/Users/xiaofeng/Code/Github/dataset/formula/enhance/img_ori'
    if not os.path.exists(IMAGE_DIR_ENHANCE):
        os.makedirs(IMAGE_DIR_ENHANCE)
# =================================远程使用数据库=========================#
elif not LOCAL and GENERATE and not ENHANCE:
    DATA_ROOT = '/home/xiaofeng/data/formula/generate/prepared/'
    MODEL_SAVED = '/home/xiaofeng/data/formula/generate/model_saved_remote/'
    # MODEL_SAVED = '/home/xiaofeng/data/formula/generate/model_saved_remote_another/'
    PREDICT_PATH = '/home/xiaofeng/data/formula/generate/predict_remote/'

# =============================远程，数据库，增强===========================#
elif not LOCAL and GENERATE and ENHANCE:
    # 数据增强参数
    FORMULA_TXT = '/home/xiaofeng/data/formula/generate_enhance/ori/formula.txt'
    NAME_TXT = '/home/xiaofeng/data/formula/generate_enhance/ori/name.txt'
    DATASET_FILE = "/home/xiaofeng/data/formula/generate_enhance/ori/im2latex.lst"
    NEW_FORMULA_FILE = "/home/xiaofeng/data/formula/generate_enhance/ori/im2latex_formulas.lst"
    TRAIN_LIST = '/home/xiaofeng/data/formula/generate_enhance/ori/train.list'
    VALIDATE_LIST = '/home/xiaofeng/data/formula/generate_enhance/ori/validate.list'
    IMAGE_DIR = '/home/xiaofeng/data/formula/generate_enhance/ori/img_ori'
    # enhance
    DATASET_FILE_ENHANCE = "/home/xiaofeng/data/formula/generate_enhance/ori/im2latex_enhance.lst"
    NEW_FORMULA_FILE_ENHANCE = "/home/xiaofeng/data/formula/generate_enhance/ori/im2latex_formulas_enhance.lst"
    NORM_FORMULA_FILE_ORI = '/home/xiaofeng/data/formula/generate_enhance/ori/formulas.norm_ori.lst'
    NORM_FORMULA_FILE_ENHANCE = '/home/xiaofeng/data/formula/generate_enhance/ori/formulas.norm_enhance.lst'
    TRAIN_LIST_ENHANCE = '/home/xiaofeng/data/formula/generate_enhance/ori/train_enhance.list'
    VALIDATE_LIST_ENHANCE = '/home/xiaofeng/data/formula/generate_enhance/ori/validate_enhance.list'
    IMAGE_DIR_ENHANCE = '/home/xiaofeng/data/formula/generate_enhance/ori/img_ori'
    if not os.path.exists(IMAGE_DIR_ENHANCE):
        os.makedirs(IMAGE_DIR_ENHANCE)
    # 模型训练参数
    DATA_ROOT = '/home/xiaofeng/data/formula/generate_enhance/prepared/'
    MODEL_SAVED = '/home/xiaofeng/data/formula/generate_enhance/model_saved_remote/'
    PREDICT_PATH = '/home/xiaofeng/data/formula/generate_enhance/predict_remote/'
    SIZE_LIST = [(120, 50), (160, 40), (240, 40), (320, 50), (360, 60),
                 (200, 40), (200, 50), (500, 100), (240, 50), (320, 40),
                 (280, 50), (280, 40), (360, 100), (400, 50), (360, 50), (360,
                                                                          40)]
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
if not os.path.exists(PREDICT_PATH):
    os.makedirs(PREDICT_PATH)
if not os.path.exists(MODEL_SAVED):
    os.makedirs(MODEL_SAVED)
SIZE_LIST = [(120, 50), (160, 40), (240, 40), (320, 50), (360, 60), (200, 40),
             (200, 50), (500, 100), (240, 50), (320, 40), (280, 50), (280, 40),
             (360, 100), (400, 50), (360, 50), (360, 40)]
SET_LIST = ['train', 'validate']
VOCAB_PATH = DATA_ROOT + 'latex_vocab.txt'
FORMULA_PATH = DATA_ROOT + 'formulas.norm.lst'
IMG_DATA_PATH = DATA_ROOT + 'images_processed/'
IMG_TEST_PATH = '/Users/xiaofeng/Code/Github/graphic/Character_mathjax_ocr/test_img/'
CHECKPOINT_PATH = MODEL_SAVED + 'ckpt/'
SUMMARY_PATH = MODEL_SAVED + 'log/'
PROPERTIES = DATA_ROOT + 'properties.npy'
NET_LIST = ['vgg_16', 'vgg_19', 'resnet_v2_50', 'resnet_v2_152']
OPTIMIZER = [
    'adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'rmsprop', 'sgd'
]
LEARNING_STYLE = ['exponential', 'fixed', 'polynomial']


def exist_or_not(pathlist):
    for path in pathlist:
        if not os.path.exists(path):
            os.makedirs(path)


exist_or_not([SUMMARY_PATH, CHECKPOINT_PATH, PRETRAINED])

# 根据词表的数量确定
V_OUT = 3 + len(open(VOCAB_PATH).readlines())
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

DISPLAY_NUMS = 10
SAVED_NUMS = 50
DECAY_STEPS = 10000
DECAY_RATE = 0.7
STAIRCASE = True
