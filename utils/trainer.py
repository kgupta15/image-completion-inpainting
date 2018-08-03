#!/usr/bin/env python

import os
from functools import reduce
import shutil

from tensorboard_logger import configure, log_value
import numpy as np
import torch
import torch.nn as nn
import cv2

from .meter import AverageMeter
from .visualizer import Visualizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    def __init__(self, title=None, config=None, data=None, models=None):
        super(Trainer, self).__init__()
        self.title = title
        self.config = config
        self.data = data

        # logs
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        self.train_loss = 0

        # parameters / model
        self.completion_net, self.local_disc, self.global_disc, self.concat_layer = models
        self.trainable_parameters = 0
        self.completed_epochs = 0
        self.cn_lr = self.config.hyperparameters['cn_lr']

        self.cn_criterion = None
        self.cn_optimizer = None
        self.ld_criterion = None
        self.ld_optimizer = None
        self.gd_criterion = None
        self.gd_optimizer = None
        self.cl_criterion = None
        self.cl_optimizer = None
        self.curr_lr = 0

        # visualization config
        # self.visualizer = Visualizer(title=self.title)

    def setName(self, name):
        self.name = name
        return True

    def setConfig(self, config):
        self.config = config
        return True

    def setData(self, data):
        self.data = data
        return True

    def setModel(self, model):
        self.model = model
        self.count_parameters()
        return True

    def setCriterion(self, criterion):
        self.cn_criterion = criterion
        self.ld_criterion = criterion
        self.gd_criterion = criterion
        self.cl_criterion = criterion
        return True

    def setCNCriterion(self, criterion):
        self.cn_criterion = criterion
        return True

    def setDiscCriterion(self, criterion):
        self.ld_criterion = criterion
        self.gd_criterion = criterion
        self.cl_criterion = criterion
        return True

    def setOptimizer(self, optimizer):
        self.cn_optimizer = optimizer
        self.ld_optimizer = optimizer
        self.gd_optimizer = optimizer
        self.concat_layer = optimizer
        return True

    def setCNOptimizer(self, optimizer):
        self.cn_optimizer = optimizer
        return True

    def setDiscOptimizer(self, optimizer):
        self.ld_optimizer = optimizer
        self.gd_optimizer = optimizer
        self.cl_optimizer = optimizer
        return True

    def setLDOptimizer(self, optimizer):
        self.ld_optimizer = optimizer
        return True

    def setGDOptimizer(self, optimizer):
        self.gd_optimizer = optimizer
        return True

    def setCLOptimizer(self, optimizer):
        self.cl_optimizer = optimizer
        return True

    def count_parameters(self):
        if self.model is None:
            raise ValueError('[-] No model has been provided')

        self.trainable_parameters = sum(reduce( lambda a, b: a*b, x.size()) for x in self.model.parameters())

    def getTrainableParameters(self):
        if self.model is not None and self.trainable_parameters == 0:
            self.count_parameters()

        return self.trainable_parameters

    def step(self):
        pass

    def save_checkpoint(self, state, is_best, checkpoint=None):
        if checkpoint is None:
            ckpt_path = os.path.join(self.config.checkpoints['loc'], self.config.checkpoints['ckpt_fname'])
        else:
            ckpt_path = os.path.join(self.config.checkpoints['loc'], checkpoint)
        best_ckpt_path = os.path.join(self.config.checkpoints['loc'], \
                            self.config.checkpoints['best_ckpt_fname'])
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copy(ckpt_path, best_ckpt_path)

    def load_saved_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            path = os.path.join(self.config.checkpoints['loc'], \
                    self.config.checkpoints['ckpt_fname'])
        else:
            path = os.path.join(self.config.checkpoints['loc'], checkpoint)
        torch.load(path)

        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("[#] Loaded Checkpoint '{}' (epoch {})"
            .format(self.config.checkpoints['ckpt_fname'], checkpoint['epoch']))
        return (start_epoch, best_prec1)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.curr_cn_lr = self.config.hyperparameters['cn_lr'] * (self.config.hyperparameters['lr_decay'] ** (epoch // self.config.hyperparameters['lr_decay_epoch']))
        for param_group in self.cn_optimizer.param_groups:
            param_group['cn_lr'] = self.curr_lr

    # def croppedImage(self):
        # i,j = np.where(mask)
        # indices = np.meshgrid(np.arange(min(i), max(i)+1), np.arange(min(j), max(j)+1), indexing='ij')
        # sub_mask = mask[indices]

    def train(self, epoch):
        if self.completion_net is None and self.local_disc is None and self.global_disc is None and self.concat_layer is None:
            raise ValueError('[-] Models have not been provided')
        if self.config is None:
            raise ValueError('[-] No Configurations present')
        if self.cn_criterion is None or self.ld_criterion is None or self.gd_criterion is None or self.cl_criterion is None:
            raise ValueError('[-] Loss Function hasn\'t been mentioned for the models')
        if self.cn_optimizer is None or self.ld_optimizer is None or self.gd_optimizer is None or self.cl_optimizer is None:
            raise ValueError('[-] Optimizer hasn\'t been mentioned for the models')
        if self.data is None:
            raise ValueError('[-] No Data available to train on')

        self.train_loss = 0
        
        # training mode
        self.completion_net.train()
        self.local_disc.train()
        self.global_disc.train()
        self.concat_layer.train()

        for batch_idx, data in enumerate(self.data):
            faces, masks, bboxes = data

            input_4ch = torch.cat([faces * (1-masks), masks], dim=1)
            assert input_4ch.shape == (faces.shape[0], faces.shape[1] + masks.shape[1], faces.shape[2], faces.shape[3])
            
            if self.config.gpu:
                faces = faces.to(device)
                masks = masks.to(device)
                input_4ch = input_4ch.to(device)

            cn_output = self.completion_net(input_4ch)
            cn_loss = self.cn_criterion(cn_output, faces)
            cn_loss.update(cn_loss.data[0], )

            self.cn_optimizer.zero_grad()
            





            cn_loss.backward()
            self.train_loss = loss.item()
            self.optimizer.step()

            log_value('train_loss', loss.item())
            if batch_idx % self.config.logs['log_interval'] == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f}'.format(
                    epoch+1, batch_idx * len(faces), len(self.data.dataset),
                    100. * batch_idx / len(self.data),
                    loss.item() / len(self.data), self.curr_lr)
                )



        # self.visualizer.add_values(epoch, loss_train=self.train_loss)
        # self.visualizer.redraw()
        # self.visualizer.block()