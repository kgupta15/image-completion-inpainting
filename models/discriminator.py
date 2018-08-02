#!/usr/bin/env python

import torch
import torch.nn as nn

class LocalDescriminator(nn.Module):
    """
    Local Descriminator
    ===================
    """
    def __init__(self):
        super(LocalDescriminator, self).__init__()
        self.relu_op = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(2,2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(2,2), bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=(2,2), bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5,5), stride=(2,2), bias=True)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5,5), stride=(2,2), bias=True)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(in_features=512*(5*5), out_features=1024, bias=True)

    def forward(self, x):
        # 64 channels
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu_op(out)
        
        # 128 channels
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu_op(out)
        
        # 256 channels
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu_op(out)
        
        # 512 channels
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu_op(out)
        out = self.conv5(out)
        out = self.bn4(out)
        out = self.relu_op(out)
        out = self.fc(out)

        return out


class GlobalDescriminator(nn.Module):
    """
    Global Descriminator
    ====================
    """
    def __init__(self):
        super(GlobalDescriminator, self).__init__()
        self.relu_op = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(2,2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), stride=(2,2), bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), stride=(2,2), bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5,5), stride=(2,2), bias=True)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5,5), stride=(2,2), bias=True)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5,5), stride=(2,2), bias=True)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(in_features=512*(5*5), out_features=1024, bias=True)

    def forward(self, x):
        # 64 channels
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu_op(out)
        
        # 128 channels
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu_op(out)
        
        # 256 channels
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu_op(out)
        
        # 512 channels
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu_op(out)
        out = self.conv5(out)
        out = self.bn4(out)
        out = self.relu_op(out)
        out = self.conv6(out)
        out = self.bn4(out)
        out = self.relu_op(out)
        out = self.fc(out)

        return out


class Concatenator(nn.Module):
    """
    Concatenator
    ============
    """
    def __init__(self, config):
        super(Concatenator, self).__init__()
        self.config = config
        self.fc = nn.Linear(in_features=2048, out_features=1, bias=True)

    def forward(self, local_des, global_des):
        out = torch.cat((local_des, global_des), 1)
        out = self.fc(out)

        return out
