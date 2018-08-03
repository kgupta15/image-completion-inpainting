#!/usr/bin/env python

import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import time

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataloader import *
from models import *
from utils import *

best_prec1 = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = NetworkConfig(args.config)

    args.distributed = config.distributed['world_size'] > 1
    if args.distributed:
        print('[+] Distributed backend')
        dist.init_process_group(backend=config.distributed['dist_backend'], init_method=config.distributed['dist_url'],\
                                world_size=config.distributed['world_size'])

    # creating models' instance
    completion_net = DilatedCompletionNetwork(config)
    local_disc = LocalDiscriminator(config)
    global_disc = GlobalDiscriminator(config)
    concat_layer = Concatenator(config)
    
    # plotting interactively
    plt.ion()

    if args.distributed:
        completion_net.to(device)
        local_disc.to(device)
        global_disc.to(device)
        concat_layer.to(device)
        completion_net = nn.parallel.DistributedDataParallel(completion_net)
        local_disc = nn.parallel.DistributedDataParallel(local_disc)
        global_disc = nn.parallel.DistributedDataParallel(global_disc)
        concat_layer = nn.parallel.DistributedDataParallel(concat_layer)

    elif config.gpu:
        completion_net = nn.DataParallel(completion_net).to(device)
        local_disc = nn.DataParallel(local_disc).to(device)
        global_disc = nn.DataParallel(global_disc).to(device)
        concat_layer = nn.DataParallel(concat_layer).to(device)

    else: return

    # Data Loading
    train_dataset = CelebDataset(args=config.data,
                                train=True,
                                transform=transforms.ToTensor())

    test_dataset = CelebDataset(args=config.data,
                                              train=False,
                                              transform=transforms.ToTensor())

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data['batch_size'], shuffle=config.data['shuffle'],
        num_workers=config.data['workers'], pin_memory=config.data['pin_memory'], sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data['batch_size'], shuffle=config.data['shuffle'],
        num_workers=config.data['workers'], pin_memory=config.data['pin_memory'])

    # Training and Evaluation
    trainer = Trainer('Image Inpainting', config, train_loader, (completion_net, local_disc, global_disc, concat_layer))
    evaluator = Evaluator('Image Inpainting Eval', config, val_loader, (completion_net, local_disc, global_disc, concat_layer))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    cn_optimizer = torch.optim.Adam(completion_net.parameters(), config.hyperparameters['cn_lr'])
    ld_optimizer = torch.optim.Adam(local_disc.parameters(), config.hyperparameters['ld_lr'])
    gd_optimizer = torch.optim.Adam(global_disc.parameters(), config.hyperparameters['gd_lr'])
    cl_optimizer = torch.optim.Adam(concat_layer.parameters(), config.hyperparameters['cl_lr'])

    trainer.setCriterion(criterion)
    trainer.setCNOptimizer(cn_optimizer)
    trainer.setLDOptimizer(ld_optimizer)
    trainer.setGDOptimizer(gd_optimizer)
    trainer.setCLOptimizer(cl_optimizer)
    evaluator.setCriterion(criterion)

    # optionally resume from a checkpoint
    if args.resume:
        trainer.load_saved_checkpoint(checkpoint=None)

    # Turn on benchmark if the input sizes don't vary
    # It is used to find best way to run models on your machine
    cudnn.benchmark = True

    best_prec1 = 0
    print('Total Epochs to be run are :{}'.format(config.hyperparameters['num_epochs']))
    for epoch in range(config.hyperparameters['num_epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        trainer.adjust_learning_rate(epoch)
        trainer.train(epoch)

        prec1 = evaluator.evaluate(epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        trainer.save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disentangling Variations', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--gpu', type=int, default=0, \
                        help="Turn ON for GPU support; default=0")
    parser.add_argument('--resume', type=int, default=0, \
                        help="Turn ON to resume training from latest checkpoint; default=0")
    parser.add_argument('--checkpoints', type=str, default="./checkpoints", \
                        help="Mention the dir that contains checkpoints")
    parser.add_argument('--config', type=str, required=True, \
                        help="Mention the file to load required configurations of the model")
    parser.add_argument('--seed', type=int, default=100, \
                        help="Seed for random function, default=100")
    parser.add_argument('--pretrained', type=int, default=0, \
                        help="Turn ON if checkpoints of model available in /checkpoints dir")
    parser.add_argument('--evaluate', type=int, default=0, \
                        help='evaluate model on validation set')
    args = parser.parse_args()

    main(args)
