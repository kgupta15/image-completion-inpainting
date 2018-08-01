#!/usr/bin/env python

import torch
import torch.nn as nn

# in_channels = 4; concat(input_image, mask)
class CompletionNetwork(nn.Module):
    """
    Completion Network
    ==================
    """
    def __init__(self, config):
        super(CompletionNetwork, self).__init__()
        self.config = config

        self.relu_op = nn.ReLU()
        self.sig_op = nn.Sigmoid()

        # 64 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(1,1), dilation=1, bias=False)

        # 128 channels
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), dilation=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)

        # 256 channels
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(2,2), dilation=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)
        self.conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)

        # 128 channels
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4), stride=(0.5,0.5), dilation=1, bias=True)
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)

        # [64,32,3] channels
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=(0.5,0.5), dilation=1, bias=True)
        self.conv14 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)
        self.output_layer = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu_op(out)
        out = self.conv2(out)
        out = self.relu_op(out)
        out = self.conv3(out)
        out = self.relu_op(out)
        out = self.conv4(out)
        out = self.relu_op(out)
        out = self.conv5(out)
        out = self.relu_op(out)
        out = self.conv6(out)
        out = self.relu_op(out)
        out = self.conv7(out)
        out = self.relu_op(out)
        out = self.conv8(out)
        out = self.relu_op(out)
        out = self.conv9(out)
        out = self.relu_op(out)
        out = self.conv10(out)
        out = self.relu_op(out)
        out = self.conv11(out)
        out = self.relu_op(out)
        out = self.conv12(out)
        out = self.relu_op(out)
        out = self.deconv1(out)
        out = self.relu_op(out)
        out = self.conv13(out)
        out = self.relu_op(out)
        out = self.deconv2(out)
        out = self.relu_op(out)
        out = self.conv14(out)
        out = self.relu_op(out)
        out = self.output_layer(out)
        out = self.sig_op(out)

        return out


class DilatedCompletionNetwork(nn.Module):
    """
    Dilated Completion Network
    ==========================
    """
    def __init__(self, config):
        super(DilatedCompletionNetwork, self).__init__()
        self.config = config

        self.relu_op = nn.ReLU()
        # 64 channels
        conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5), stride=(1,1), dilation=1, bias=False)

        # 128 channels
        conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), dilation=1, bias=False)
        conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), dilation=1, bias=False)

        # 256 channels
        conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(2,2), dilation=1, bias=False)
        conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=False)
        conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=False)
        conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=2, bias=False)
        conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=4, bias=False)
        conv9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=8, bias=False)
        conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=16, bias=False)
        conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=False)
        conv12 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), dilation=1, bias=False)

        # 128 channels
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4), stride=(0.5,0.5), dilation=1, bias=True)
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)

        # [64,32,3] channels
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=(0.5,0.5), dilation=1, bias=True)
        self.conv14 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)
        self.output_layer = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3,3), stride=(1,1), dilation=1, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu_op(out)
        out = self.conv2(out)
        out = self.relu_op(out)
        out = self.conv3(out)
        out = self.relu_op(out)
        out = self.conv4(out)
        out = self.relu_op(out)
        out = self.conv5(out)
        out = self.relu_op(out)
        out = self.conv6(out)
        out = self.relu_op(out)
        out = self.conv7(out)
        out = self.relu_op(out)
        out = self.conv8(out)
        out = self.relu_op(out)
        out = self.conv9(out)
        out = self.relu_op(out)
        out = self.conv10(out)
        out = self.relu_op(out)
        out = self.conv11(out)
        out = self.relu_op(out)
        out = self.conv12(out)
        out = self.relu_op(out)
        out = self.deconv1(out)
        out = self.relu_op(out)
        out = self.conv13(out)
        out = self.relu_op(out)
        out = self.deconv2(out)
        out = self.relu_op(out)
        out = self.conv14(out)
        out = self.relu_op(out)
        out = self.output_layer(out)
        out = self.sig_op(out)

        return out
