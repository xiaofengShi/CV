# -*- coding: utf-8 -*-
"""
识别字体比例: 100x160


字号与偏移的对应关系：

字号 -- 汉字偏移  -- 字母符号数字偏移
82     [0, -16]     [2, -22]
80     [0, -16]     [2, -22]
74     [0, -16]     [2, -21]
68     [0, -16]     [2, -19]
62     [0, -14]     [2, -16]
56     [0, -12]     [2, -14]
50     [0, -10]     [2, -12]
40     [0, -10]     [2, -10]
36     [0, -8]      [2, -10]
32     [0, -7]      [2, -9]
28     [0, -6]      [0, -9]
24     [0, -5]      [0, -6]
20     [0, -4]      [0, -5]
"""



import codecs
import os
from PIL import Image, ImageFont, ImageDraw
import uuid
import numpy as np


def image_crop(img):
    try:
        size, _ = img.size
        data = np.array(img)
        points = [i for i in range(size) if np.sum(data[:, i, :]) < 255*3*size]
        start = points[0]
        end = points[-1]
        img_ = img.crop([start, 0, end+1, size])
    except:
        pass
    return img_



def gen_image(chars, size, font, dir, h_offset=0, v_offset=0):

    if not os.path.exists(dir):
        os.makedirs(dir)

    font = ImageFont.truetype(font, size)
    for text in chars:
        im = Image.new("RGB", (size, size), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        dr.text((h_offset, v_offset), text, font=font, fill="#000000")
        im = image_crop(im)

        path = os.path.join(dir, f"{text}{uuid.uuid1()}.png")
        im.save(path, format="png")


def test_gen_image(text, size, font, dir, h_offset=0, v_offset=0):

    if not os.path.exists(dir):
        os.makedirs(dir)

    im = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(font, size)
    dr.text((h_offset, v_offset), text, font=font, fill="#000000")
    im = image_crop(im)
    path = os.path.join(dir, f"{text}{uuid.uuid1()}.png")
    im.save(path, format="png")

if __name__ == '__main__':
    font = "E:/Mory/gogs/tensorflow-ocr/fonts/sim/FanZhenCuShongJianTi.ttf"
    size = 26
    dir = f"data/special/{size}/"
    downs = "﹄﹂"

    gen_image(downs, size, font, dir, 0, 5)


    ups = "﹃﹁"
    gen_image(ups, size, font, dir, 0, -10)
