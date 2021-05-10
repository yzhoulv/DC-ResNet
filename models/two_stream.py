# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

@author: zain
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from config.config import Config as opt
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.channel = ChannelBlock(planes)
            self.spacial = SpacialBlock(kernel_size=3)

    def forward(self, input_value):
        x, weights = input_value
        if weights is None:
            weights = []
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        weight_c = None
        weight_s = None
        if self.use_se:
            out, weight_c = self.channel(out)
            out, weight_s = self.spacial(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)
        weights.append(weight_c)
        weights.append(weight_s)
        return out, weights


class AttentionDrop(nn.Module):

    def __init__(self, kernel_size=3):
        super(AttentionDrop, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.p = 0.8  # 越小遮的越少

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_con = torch.cat([avg_out, max_out], dim=1)
        x_con = self.conv1(x_con)
        x_con_sq = torch.squeeze(x_con)
        x_norm = x_con_sq
        x_norm = x_norm.reshape(x_norm.shape[0], x_norm.shape[1] * x_norm.shape[2])
        vals, _ = x_norm.topk(int((1 - self.p) * x_norm.shape[1]), dim=1, sorted=False)
        threholds, _ = torch.min(vals, dim=1, keepdim=True)
        threholds = threholds.expand(threholds.shape[0], x.shape[2] * x.shape[3])
        mask = torch.where(x_norm - threholds < 0, torch.ones(x_norm.shape, device="cuda"),
                           torch.zeros(x_norm.shape, device="cuda"))
        mask = mask.reshape(mask.shape[0], x_con_sq.shape[1], x_con_sq.shape[2])
        x_iptc = self.sigmoid(x_con_sq)
        x_rand = torch.rand(size=x_iptc.shape).cuda()
        x_rand = torch.where(x_rand - 0.5 < 0, torch.ones(x_rand.shape, device="cuda"),
                             torch.zeros(x_rand.shape, device="cuda"))
        final_map = mask * x_rand * x_iptc + x_iptc * (1.0 - x_rand)
        final_map = final_map.unsqueeze(1)

        return x * final_map


class SpacialBlock(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpacialBlock, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y, y


class ChannelDropout(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelDropout, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.p = 0.5

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = y_max + y_avg
        y = self.fc(y)
        vals, _ = y.topk(int((1 - self.p) * y.shape[1]), dim=1, sorted=False)
        threholds, _ = torch.min(vals, dim=1, keepdim=True)
        # print(threholds.shape)
        threholds = threholds.expand(threholds.shape[0], y.shape[1])
        # threholds = threholds.view(b, c, 1, 1)
        mask = torch.where(y - threholds < 0, torch.ones(y.shape, device="cuda"),
                           torch.zeros(y.shape, device="cuda"))
        # mask = mask.reshape(mask.shape[0], y.shape[1], y.shape[2])
        rand = torch.rand(size=y.shape).cuda()
        rand = torch.where(rand - 0.5 < 0, torch.ones(rand.shape, device="cuda"),
                             torch.zeros(rand.shape, device="cuda"))
        final_map = mask * rand * y + y * (1.0 - rand)
        final_map = final_map.view(b, c, 1, 1)
        return x * final_map


class ChannelBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = y_max + y_avg
        y = self.fc(y).view(b, c, 1, 1)
        return x * y, y


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_max = self.max_pool(x).view(b, c)
        y = y_max + y_avg
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        # self.drop = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(512 * 8 * 8, 512)
        # self.bn5 = nn.BatchNorm1d(512)
        # self.fc2 = SoftMaxProduct(512, opt.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        weight = []
        x = self.conv(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x, weight = self.layer1((x, weight))
        x, weight = self.layer2((x, weight))
        x, weight = self.layer3((x, weight))
        x, weight = self.layer4((x, weight))
        x = self.bn4(x)
        # x = x.view(x.size(0), -1)
        # x = self.drop(x)
        # x = self.fc1(x)
        # feat = self.bn5(x)
        # cal_score = self.fc2(feat)
        return x, weight


class TwoStream(nn.Module):
    def __init__(self, block, layers, use_se=True):
        super(TwoStream, self).__init__()
        self.model_h = ResNetFace(block, layers, use_se=use_se)
        self.model_l = ResNetFace(block, layers, use_se=use_se)
        self.layer_h = self._last_layer(512, 8)
        self.layer_l = self._last_layer(512, 8)
        self.layer_hl = self._last_layer(512, 8)
        self.adrop = ChannelDropout(512)

    def _cal_loss(self, weight_h, weight_l):
        weight_loss = 0
        for i in range(len(weight_l)):
            weight_loss += torch.norm(weight_h[i] - weight_l[i])
        return weight_loss / len(weight_l)

    def _last_layer(self, channel, size):
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(channel * size * size, 512),
            nn.BatchNorm1d(512)
        )

    def forward(self, x_h, x_l):
        x_h, weights_h = self.model_h(x_h)
        x_l, weights_l = self.model_l(x_l)
        x_hl = self.adrop(x_h)
        x_h = x_h.view(x_h.size(0), -1)
        x_l = x_l.view(x_l.size(0), -1)
        x_hl = x_hl.view(x_hl.size(0), -1)
        cal_score_h = self.layer_h(x_h)
        cal_score_l = self.layer_l(x_l)
        cal_score_hl = self.layer_hl(x_hl)
        res_loss = torch.norm(cal_score_l - cal_score_hl) / 256
        weights = self._cal_loss(weights_h, weights_l)
        return cal_score_h, cal_score_l, (weights, res_loss)


def resnet_face18(use_se=True, pretrained=False, **kwargs):
    model = TwoStream(IRBlock, [2, 3, 2, 2], use_se=use_se, **kwargs)
    if pretrained:
        # model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("./output/my_pre_new.pth"))
    return model
