# coding=utf8
'''
File: own_annotation.py
Project: dataset_tools
File Created: Sunday, 2nd September 2018 5:14:30 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Sunday, 2nd September 2018 5:15:40 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''
""" create annotation use own dataset """

import json
import os
import glob
import random
from collections import OrderedDict

# classes = ['height', 'width', 'name', 'formula', 'text', 'graphic', 'no_gene']
classes = ['formula', 'text', 'graphic', 'no_gene']


def read_label(label_path, list_file):
    label_name = os.path.splitext(os.path.basename(label_path))[0]
    f = open(label_path, 'r', encoding='utf-8')
    label_dict = json.load(f)
    assert label_name == label_dict['name'], 'Label file name must be same with the img name'
    for obj in classes:
        obj_contens = label_dict[obj]
        if obj_contens:
            cla_ind = classes.index(obj)
            for box in obj_contens:
                box_want = [int(i) for i in box]
                list_file.write(" " + ",".join([str(a) for a in box_want]) + ',' + str(cla_ind))


def main(dataset):
    for data in dataset:
        if os.path.exists(os.path.join(img_root, data+'.jpg')):
            # img_path = os.path.join(img_root, data + '.jpg')
            img_path = data + '.jpg'
        else:
            # img_path = os.path.join(img_root, data + '.png')
            img_path = data + '.png'
        list_file.write(img_path)
        label_path = os.path.join(label_root, data + '.json')
        read_label(label_path, list_file)
        list_file.write('\n')


if __name__ == '__main__':
    img_root = '/Users/xiaofeng/Desktop/images_4_color'
    label_root = '/Users/xiaofeng/Desktop/Annotations'
    img_list = [os.path.splitext(img)[0] for img in os.listdir(
        img_root) if img.endswith('.png') or img.endswith('.jpg')]
    label_list = [os.path.splitext(label)[0]
                  for label in os.listdir(label_root) if label.endswith('.json')]
    union_dataset = [data for data in img_list if data in label_list]
    print('union_dataset:', len(union_dataset))
    perception = 0.9
    random.shuffle(union_dataset, random.seed(9001))
    dataset = OrderedDict()
    dataset['train'] = union_dataset[:int(perception * len(union_dataset))]
    dataset['val'] = union_dataset[int(perception * len(union_dataset)):]
    for set in dataset.keys():
        dataset_set = dataset[set]
        print('{}-nums:[{}]'.format(set, len(dataset_set)))
        list_file = open('../data/' + set + '.txt', 'w')
        main(dataset_set)
        list_file.close()
