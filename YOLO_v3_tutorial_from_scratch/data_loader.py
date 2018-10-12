# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from util import prepare_dataset


class DataIter(data.Dataset):

    def __init__(self, root, transforms=None):
        '''
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        Args:
            root: 训练集和测试集的位置
            transform: 是否进行数据增强
            train: 训练标志
        '''
        self.dataset_list = root.readlines()

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            self.transforms = T.Compose([T.ToTensor(),
                                         normalize])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        single_data = self.dataset_list[index]

        data, label = prepare_dataset(single_data, (416, 416))
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.dataset_list)
