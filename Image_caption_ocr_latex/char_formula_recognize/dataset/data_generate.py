#!/usr/bin/env python
# _Author_: xiaofeng
# Date: 2018-04-13 17:20:01
# Last Modified by: xiaofeng
# Last Modified time: 2018-04-13 17:20:01
# -*- coding: utf-8 -*-
'''
连接MongoDB数据库，找到存在http网址的字符串；
将url保存并下载到本地，保存位置为'./data/image/'
图片明明规则：5a45403a8223977701b0aa6a_3.5-Y-A3-2-2-1.png
            object id_src.split('/')[-1]
可以从图片的名称中，方便的找到该图片对应的数据库中的id以及图片的url地址信息
    1.将数据库中的body进行正则匹配
    2.将匹配出来的latex格式进行txt格式存储
    3.使用latex生成pdf，在转换成png格式
'''

from pymongo import MongoClient
import re, os, sys, requests, glob
import string, copy
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)
from subprocess import call
import hashlib
from PIL import Image
import random
import tex2pix
import config as cfg
from multiprocessing import Pool
'''
预设地址，端口，数据库名称，collection名称
'''
Host = '10.8.8.71'
Port = 27017
database_name = 'knowledge_graph'
Collection_name = 'problems_info'

MIN_LENGTH = 10
MAX_LENGTH = 500
MAX_NUMBER = 150 * 1000
THREADS = 3
TRAIN_PERSP = 0.8
DEVNULL = open(os.devnull, "w")
# Running a thread pool masks debug output. Set DEBUG to 1 to run
# formulas over images sequentially to see debug errors more clearly
'''
FORMULA_TXT = './generate/formula.txt'
NAME_TXT = './generate/name.txt'
DATASET_FILE = "./generate/im2latex.lst"
NEW_FORMULA_FILE = "./generate/im2latex_formulas.lst"
TRAIN_LIST = './generate/train.list'
VALIDATE_LIST = './generate/validate.list'
IMAGE_DIR = '/Users/xiaofeng/Code/Github/dataset/formula/data_formula'
##
DATASET_FILE_ENHANCE = "./generate/im2latex_enhance.lst"
NEW_FORMULA_FILE_ENHANCE = "./generate/im2latex_formulas_enhance.lst"
NORM_FORMULA_FILE_ORI = './generate/formulas.norm_ori.lst'
NORM_FORMULA_FILE_ENHANCE = './generate/formulas.norm_enhance.lst'
TRAIN_LIST_ENHANCE = './generate/train_enhance.list'
VALIDATE_LIST_ENHANCE = './generate/validate_enhance.list'
IMAGE_DIR_ENHANCE = '/Users/xiaofeng/Code/Github/dataset/formula/enhance/img_ori'
'''

DEBUG = False
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

# 连接数据库，进行待分析位置的定位
# 创建MongoDB连接
client = MongoClient(host=Host, port=Port)
# 选择要连接的数据库名称
db = client[database_name]
# 选择当前数据库下的指定名称的collection
collections = db[Collection_name]

print('database name is %s ,collection name is %s' % (db.name,
                                                      collections.name))
RENDERING_SETUPS = {
    'basic': [
        BASIC_SKELETON, "convert -density 200 -quality 100 %s.pdf %s.png",
        lambda filename: os.path.isfile(filename + ".png")
    ]
}


def remove_temp_files(name):
    """ Removes .aux, .log, .pdf and .tex files for name """
    os.remove(name + ".aux")
    os.remove(name + ".log")
    os.remove(name + ".pdf")
    os.remove(name + ".tex")


# 通过使用网址‘http://latex.codecogs.com’来生成img，会存在很多无法识别的情况
def formula_as_file(formula, file, negate=False):
    tfile = file
    if negate:
        tfile = 'tmp.png'
    r = requests.get(
        'http://latex.codecogs.com/png.latex?\dpi{300} \huge %s' % formula)
    f = open(tfile, 'wb')
    f.write(r.content)
    f.close()
    if negate:
        os.system(
            'convert tmp.png -channel RGB -negate -colorspace rgb %s' % file)


# 进行网址信息的正则匹配
def url_exist_or_not(content):
    read = content['body']
    result = re.findall("(?isu)(http\://[a-zA-Z0-9\.\?%+-/&\=\:]+)", read)
    return result


# 进行公式的正则匹配
def formula_exist_or_not(content):
    read = content['body']
    pattern = [
        r"\\begin\{equation\}(.*?)\\end\{equation\}", r"\$\$(.*?)\$\$",
        r"\$(.*?)\$", r"\\\[(.*?)\\\]", r"\\\((.*?)\\\)"
    ]
    ret = []
    for pat in pattern:
        res = re.findall(pat, read, re.DOTALL)
        res = [
            x.strip().replace('\n', '').replace('\r', '') for x in res
            if MAX_LENGTH > len(list(set(x.strip()))) > MIN_LENGTH
        ]
        ret.extend(res)
    return ret


# 生成公式txt
def formula_txt():
    # 当前collection中包含的document数量
    with open(cfg.FORMULA_TXT, 'w') as f:
        with open(cfg.NAME_TXT, 'w') as g:
            length = collections.count()
            print('total length', length)
            number = 1
            for content in collections.find():
                # result = url_exist_or_not(content)
                result = formula_exist_or_not(content)
                if result:
                    count = 0
                    id = content['_id']
                    for res in result:
                        current_id = str(id) + '_' + str(count)
                        formula = ''.join(res)
                        formu_info = str(formula) + '\n'
                        name_info = current_id + '\n'
                        f.write(formu_info)
                        g.write(name_info)
                        count += 1
                        number += 1
    print('Foumula nums genereated is: ', number)


# generate the image based formula
def formula_to_image(formula):
    """ Turns given formula into images based on RENDERING_SETUPS
    returns list of lists [[image_name, rendering_setup], ...], one list for
    each rendering.
    Return None if couldn't render the formula"""
    formula = formula.strip("%")
    # name = hashlib.sha1(formula.encode('utf-8')).hexdigest()
    md5 = hashlib.md5()
    md5.update(formula.encode('utf-8'))
    name = md5.hexdigest()
    ret = []
    skiping = []
    for rend_name, rend_setup in RENDERING_SETUPS.items():
        full_path = name + "_" + rend_name
        if len(rend_setup) > 2 and rend_setup[2](full_path):
            print('Remake the full_path')
            full_path = name + "_" + rend_name + '_' + str(
                random.randint(0, 1000000))
            print('New full_path is :', full_path)
            # print("Skipping, already done: " + full_path)
            # ret.append([full_path, rend_name])
            # continue
        # Create latex source
        latex = rend_setup[0] % formula
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
            return None

        # Turn .pdf to .png
        # Handles variable number of places to insert path.
        # i.e. "%s.tex" vs "%s.pdf %s.png"
        full_path_strings = rend_setup[1].count("%") * (full_path, )

        code = call(
            (rend_setup[1] % full_path_strings).split(" "),
            stdout=DEVNULL,
            stderr=DEVNULL)
        #Remove files
        try:
            remove_temp_files(full_path)
        except Exception as e:
            # try-except in case one of the previous scripts removes these files
            # already
            return None

        # Detect of convert created multiple images -> multi-page PDF
        resulted_images = glob.glob(full_path + "-*")
        if code != 0:
            # Error during rendering, remove files and return None
            os.system("rm -rf " + full_path + "*")
            return None
        elif len(resulted_images) > 1:
            # We have multiple images for same formula
            # Discard result and remove files
            for filename in resulted_images:
                os.system("rm -rf " + filename + "*")
            return None
        else:
            ret.append([full_path, rend_name])

    return ret


# generate img form formulas list  and saved the dataset for model


def generate_formula_lst_img(FORMULA_TXT, IMAGE_DIR, NEW_FORMULA_FILE,
                             DATASET_FILE):
    if not os.path.exists(FORMULA_TXT):
        formula_txt()
        formulas = open(FORMULA_TXT).read().split('\n')
    else:
        formulas = open(FORMULA_TXT).read().split('\n')
    try:
        os.mkdir(IMAGE_DIR)
    except OSError as e:
        pass  #except because throws OSError if dir exists
    print("Turning formulas into images...")
    # Change to image dir because textogif doesn't seem to work otherwise...
    oldcwd = os.getcwd()
    # Check we are not in image dir yet (avoid exceptions)
    if not IMAGE_DIR in os.getcwd():
        os.chdir(IMAGE_DIR)

    names = None

    if DEBUG:
        names = [formula_to_image(formula) for formula in formulas]
    else:
        pool = Pool(THREADS)
        names = list(pool.imap(formula_to_image, formulas))
    # 切换到到当前路径
    os.chdir(oldcwd)

    zipped = list(zip(formulas, names))

    new_dataset_lines = []
    new_formulas = []
    ctr = 0
    for formula in zipped:
        if formula[1] is None:
            continue
        for rendering_setup in formula[1]:
            new_dataset_lines.append(
                str(ctr) + " " + " ".join(rendering_setup))
        new_formulas.append(formula[0])
        ctr += 1
    print('total', ctr)
    with open(NEW_FORMULA_FILE, "w") as f:
        f.write("\n".join(new_formulas))

    with open(DATASET_FILE, "w") as f:
        f.write("\n".join(new_dataset_lines))


# divide the dataset into train and test
def generate_train_test_batch(DATASET_FILE, FORMULA_TXT, IMAGE_DIR,
                              NEW_FORMULA_FILE, TRAIN_LIST, VALIDATE_LIST):
    if not os.path.exists(DATASET_FILE):
        generate_formula_lst_img(FORMULA_TXT, IMAGE_DIR, NEW_FORMULA_FILE,
                                 DATASET_FILE)
        formulas = open(DATASET_FILE).read().split('\n')
    else:
        formulas = open(DATASET_FILE).read().split('\n')
    total_length = len(formulas)
    train_length = int(TRAIN_PERSP * total_length)
    random.shuffle(formulas)
    train = formulas[:train_length]
    valid = formulas[train_length:]
    print(len(train), len(valid))
    print('creating train and test lis')
    with open(TRAIN_LIST, 'w') as tr:
        tr.write('\n'.join(train))
    with open(VALIDATE_LIST, 'w') as va:
        va.write('\n'.join(valid))


# 使用生成的标准化的公式进行公式增强
# Data enhance and create the new formula txt.
# Tips: replace the numbber(0~9) and the character(a~z A~Z)
def data_enhance(NORM_FORMULA_FILE_ORI, NORM_FORMULA_FILE_ENHANCE):
    assert os.path.exists(NORM_FORMULA_FILE_ORI)
    formulas_total = open(NORM_FORMULA_FILE_ORI).read().split('\n')
    print('Formulas nums in dataset is :', len(formulas_total))
    # number list and character list
    # list [0---9]
    number_list = list(string.digits)
    # list [a---z]
    # character_list_low = list(string.ascii_lowercase)
    character_list_low = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o',
        'p', 'q', 's', 't', 'w', 'x', 'y', 'z'
    ]
    # list [A---Z]
    # character_list_up = list(string.ascii_uppercase)
    character_list_up = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'V', 'X', 'Y', 'Z'
    ]
    ####
    # function_group_charactors = ['\begin', '\end']
    # fraction_charactors = ['\frac']
    # copy the formula_total
    generate_formulas_total = []
    generate_formulas_merge = []
    for formula in formulas_total:
        formula_list = formula.split(' ')
        # formula_back = ' '.join(i for i in formula_list)
        for char_index in range(len(formula_list)):
            char_value = formula_list[char_index]
            generate_formula_list = copy.deepcopy(formula_list)
            if char_value in number_list:
                index_in_num_list_now = number_list.index(char_value)
                if index_in_num_list_now < 1:
                    num_list_in = number_list[index_in_num_list_now:
                                              index_in_num_list_now + 1]
                elif index_in_num_list_now > len(number_list) - 1:
                    num_list_in = number_list[index_in_num_list_now - 1:
                                              index_in_num_list_now]
                else:
                    num_list_in = number_list[index_in_num_list_now - 1:
                                              index_in_num_list_now + 1]
                for new_num in num_list_in:
                    generate_formula_list[char_index] = new_num
                    generate_formula_str = ' '.join(
                        i for i in generate_formula_list)
                    generate_formulas_total.append(generate_formula_str)

                    generate_formula_str_merge = ''.join(
                        i for i in generate_formula_list)
                    generate_formulas_merge.append(generate_formula_str_merge)
            elif char_value in character_list_low:
                index_in_char_list_low = character_list_low.index(char_value)
                if index_in_char_list_low < 1:
                    char_list = character_list_low[index_in_char_list_low:
                                                   index_in_char_list_low + 1]
                elif index_in_char_list_low > len(character_list_low) - 1:
                    char_list = character_list_low[index_in_char_list_low - 1:
                                                   index_in_char_list_low]
                else:
                    char_list = character_list_low[index_in_char_list_low - 1:
                                                   index_in_char_list_low + 1]
                for new_char in char_list:
                    generate_formula_list[char_index] = new_char
                    generate_formula_str = ' '.join(
                        i for i in generate_formula_list)
                    generate_formulas_total.append(generate_formula_str)
                    generate_formula_str_merge = ''.join(
                        i for i in generate_formula_list)
                    generate_formulas_merge.append(generate_formula_str_merge)
            elif char_value in character_list_up:
                index_in_char_list_up = character_list_up.index(char_value)
                if index_in_char_list_up < 1:
                    char_list = character_list_up[index_in_char_list_up:
                                                  index_in_char_list_up + 1]
                elif index_in_char_list_up > len(character_list_up) - 1:
                    char_list = character_list_up[index_in_char_list_up - 1:
                                                  index_in_char_list_up]
                else:
                    char_list = character_list_up[index_in_char_list_up - 1:
                                                  index_in_char_list_up + 1]
                for new_char in char_list:
                    generate_formula_list[char_index] = new_char
                    generate_formula_str = ' '.join(
                        i for i in generate_formula_list)
                    generate_formulas_total.append(generate_formula_str)
                    generate_formula_str_merge = ''.join(
                        i for i in generate_formula_list)
                    generate_formulas_merge.append(generate_formula_str_merge)
            else:
                continue
    print('Formulas nums in dataset is :', len(generate_formulas_total))
    with open(NORM_FORMULA_FILE_ENHANCE, 'w') as enhance:
        enhance.write('\n'.join(generate_formulas_total))
    # with open(NORM_FORMULA_FILE_ENHANCE_MERGE, 'w') as enhance:
    #     enhance.write('\n'.join(generate_formulas_merge))
    print('Done')


if __name__ == '__main__':
    # step 1
    # enhance formula
    # 首先根据数据库中的标准化之后的公式进行数据增强
    data_enhance(cfg.NORM_FORMULA_FILE_ORI, cfg.NORM_FORMULA_FILE_ENHANCE)

    # step 2
    # 根据新生成的公式列表进行图像的生成
    generate_train_test_batch(
        cfg.DATASET_FILE_ENHANCE, cfg.NORM_FORMULA_FILE_ENHANCE,
        cfg.IMAGE_DIR_ENHANCE, cfg.NEW_FORMULA_FILE_ENHANCE,
        cfg.TRAIN_LIST_ENHANCE, cfg.VALIDATE_LIST_ENHANCE)
    sys.exit()
