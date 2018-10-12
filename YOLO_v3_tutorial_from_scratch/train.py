'''
File: train.py
Project: YOLO_v3_tutorial_from_scratch
File Created: Friday, 31st August 2018 6:15:47 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Friday, 31st August 2018 6:15:51 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''
""" reference
[1]:https://github.com/qqwweee/keras-yolo3/blob/master/train.py
[2]:https://blog.csdn.net/maweifei/article/details/81204702
[3]:https://github.com/eriklindernoren/PyTorch-YOLOv3
"""
from darknet import Darknet
from config import opt
from torchnet import meter

import torch as t
from data_loader import DataIter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer
import logging
from logging.handlers import RotatingFileHandler
from visdom import Visdom


def init_logger(log_file):
    '''得到一个日志的类对象
    Args:
        log_file   :  日志文件名
    Returns:
        logger     :  日志类对象
    '''

    logger = logging.getLogger()
    hdl = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=10)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    hdl.setFormatter(formatter)
    logger.addHandler(hdl)
    logger.setLevel(logging.DEBUG)
    return logger


class Train(object):
    def __init__(self, batch_data, cfg_file):
        self.data = batch_data
        self.cfg_file = cfg_file
        self.logger = init_logger(cfg.log_file)
        self.Yolo_model = self.Yolo_model()

    def Yolo_model(self):
        Yolo_model = Darknet(self.cfg_file)
        img_size = int(Yolo_model.net_info["height"])
        assert img_size % 32 == 0 and img_size > 32
        return Yolo_model

    def train(self, model, train_loader, loss_fn, optimizer, logger, print_every=20, USE_CUDA=True):
        '''训练一个epoch，即将整个训练集跑一次
        Args:
            model         :  定义的网络模型
            train_loader  :  加载训练集的类对象
            loss_fn       :  损失函数，此处为CTCLoss
            optimizer     :  优化器类对象
            logger        :  日志类对象
            print_every   :  每20个batch打印一次loss
            USE_CUDA      :  是否使用GPU
        Returns:
            average_loss  :  一个epoch的平均loss
        '''
        model.train()

        total_loss = 0
        print_loss = 0
        i = 0
        for data in train_loader:
            inputs, targets, input_sizes, input_sizes_list, target_sizes = data
            batch_size = inputs.size(0)
            inputs = inputs.transpose(0, 1)

            inputs = Variable(inputs, requires_grad=False)
            input_sizes = Variable(input_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)
            target_sizes = Variable(target_sizes, requires_grad=False)

            if USE_CUDA:
                inputs = inputs.cuda()

            inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_sizes_list)

            out = model(inputs)
            loss = loss_fn(out, targets, input_sizes, target_sizes)
            loss /= batch_size
            print_loss += loss.data[0]

            if (i + 1) % print_every == 0:
                print('batch = %d, loss = %.4f' % (i+1, print_loss / print_every))
                logger.debug('batch = %d, loss = %.4f' % (i+1, print_loss / print_every))
                print_loss = 0

            total_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 400)
            optimizer.step()
            i += 1
        average_loss = total_loss / i
        print("Epoch done, average loss: %.4f" % average_loss)
        logger.info("Epoch done, average loss: %.4f" % average_loss)
        return average_loss

    def trainner(self, model):

        seed = torch.cuda.initial_seed()

        # 1. data
        train_data = DataIter(opt.train_data_root)
        val_data = DataIter(opt.test_data_root)
        train_dataloader = DataLoader(train_data, opt.batch_size,
                                      shuffle=True, num_workers=opt.num_workers)
        val_dataloader = DataLoader(val_data, opt.batch_size,
                                    shuffle=False, num_workers=opt.num_workers)

        # 2.configure model
        if opt.load_model_path:
            model.load_state_dict(torch.load(opt.load_model_path))

        self.logger.info("Model Structure:")
        for idx, m in enumerate(model.children()):
            print(idx, m)
            self.logger.info(str(idx) + "->" + str(m))

        # criterion and optimizer
        criterion = t.nn.CrossEntropyLoss()
        lr = opt.lr
        optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

        torch.manual_seed(seed)
        if opt.use_gpu:
            model.cuda()
            criterion.cuda()
        # 统计混淆矩阵
        loss_meter = meter.AverageValueMeter()

        params = {
            'num_epoches': opt.max_epoch, 'end_adjust_acc': opt.end_adjust_acc, 'seed': seed,
            'lr_decay': opt.lr_decay, 'learning_rate': opt.lr, 'weight_decay': opt.weight_decay,
            'batch_size': opt.batch_size}

        print(params)

        # visualization for training
        viz = Visdom()
        title = 'TIMIT YOLO_v3 Acoustic Model'

        opts = [dict(title=title+" Loss", ylabel='Loss', xlabel='Epoch'),
                dict(title=title+" Loss on Dev", ylabel='DEV Loss', xlabel='Epoch'),
                dict(title=title+' CER on DEV', ylabel='DEV CER', xlabel='Epoch')]
        viz_window = [None, None, None]

        count = 0
        learning_rate = opt.lr
        loss_best = 1000
        loss_best_true = 1000
        adjust_rate_flag = False
        stop_train = False
        adjust_time = 0
        acc_best = 0
        start_time = time.time()
        loss_results = []
        dev_loss_results = []
        dev_cer_results = []

        while not stop_train:
            if count >= opt.max_epoch:
                break
            count += 1

            if adjust_rate_flag:
                learning_rate *= opt.lr_decay
                adjust_rate_flag = False
                for param in optimizer.param_groups:
                    param['lr'] *= opt.lr_decay
                print("Start training epoch: %d, learning_rate: %.5f" % (count, learning_rate))
                self.logger.info("Start training epoch: %d, learning_rate: %.5f" %
                                 (count, learning_rate))
