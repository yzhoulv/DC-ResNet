#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import Parameter
from config.config import Config as opt

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.inplans = inplanes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class DropoutFC(nn.Module):
    def __init__(self):
        super(DropoutFC, self).__init__()
        if self.training:
            self.p = 0.5
        else:
            self.p = 0.0
        # self.sav = torch.zeros(960, 8, 8).cuda()
        # nn.init.xavier_uniform_(self.sav)
        self.sav = Parameter(torch.FloatTensor(960, 8, 8))
        nn.init.xavier_uniform_(self.sav)
        self.dropout = nn.Dropout(p=self.p)
        self.fc = nn.Linear(960, 659)

    def forward(self, x):
        x = x * self.sav
        x = x.sum((2, 3)).view(x.shape[0], 960)
        feat = self.dropout(x)
        cal_class = self.fc(feat)
        return cal_class, cal_class


class Led3D(nn.Module):

    def __init__(self, basic_block):
        super(Led3D, self).__init__()
        self.layer1 = self._make_layer(basic_block, 3, 32, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=33, stride=16, padding=16)
        self.layer2 = self._make_layer(basic_block, 32, 64, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=17, stride=8, padding=8)
        self.layer3 = self._make_layer(basic_block, 64, 128, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=9, stride=4, padding=4)
        self.layer4 = self._make_layer(basic_block, 128, 256, stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer5 = self._make_layer(basic_block, 480, 960, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc = DropoutFC()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, outplanes, stride=1):
        return block(inplanes, outplanes, stride)

    def forward(self, x):
        if x.shape[1] == 4:
            # depth = x[:, 3:, :, :]
            x = x[:, 0:3, :, :]
        x = self.layer1(x)
        x_m1 = self.maxpool1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x_m2 = self.maxpool2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x_m3 = self.maxpool3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x_m4 = self.maxpool4(x)
        x = torch.cat((x_m1, x_m2, x_m3, x_m4), dim=1)
        x = self.layer5(x)
        feat, cls_score = self.fc(x)
        return feat, cls_score


def Led3DNet(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Led3D(BasicBlock, **kwargs)
    if pretrained:
        # model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("./output/cur_lock_pretrain_new.pth"))
        model.fc.fc = nn.Linear(960, 52)

    return model


if __name__ == '__main__':
    model = Led3D(BasicBlock)
    model.train()
    input = torch.randn(10, 4, 128, 128)
    feat, cls_score = model(input)
    print(feat.shape, cls_score.shape)
