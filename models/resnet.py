# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

@author: ronghuaiyang
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
from torch.nn import Parameter
from config.config import Config as opt

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SoftMaxProduct(nn.Module):
    r"""Implement of softmax:
        Args:
            in_features: size of each input sample
            out_features: size of each output sample

        """

    def __init__(self, in_features, out_features):
        super(SoftMaxProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        output = F.linear(input, self.weight)
        return output


class MyLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=True):
        super(MyLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _weight_loss(self, weight, label):

        weight = F.normalize(weight, dim=1)
        weight_label = weight[label]
        temp_res = weight_label.repeat(1, weight.shape[0])
        temp_res = temp_res.reshape(label.shape[0] * weight.shape[0], 512)
        temp_res = torch.sum((temp_res - weight.repeat(label.shape[0], 1)) ** 2, dim=1)
        temp_res = temp_res.reshape(label.shape[0], -1)
        values = torch.where(temp_res == 0, torch.ones(temp_res.shape, device="cuda")*1000, temp_res)
        value, _ = torch.min(values, dim=1)
        return label.shape[0] / (torch.sum(value))

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        weight_loss = self._weight_loss(self.weight, label)
        # weight_loss = 0

        return output, weight_loss
        # return input, 0

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AttentionDrop(nn.Module):

    def __init__(self, kernel_size=3):
        super(AttentionDrop, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.p = 0.8 # 越小遮的越少

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_con = torch.cat([avg_out, max_out], dim=1)
        x_con = self.conv1(x_con)
        x_con_sq = torch.squeeze(x_con)
        x_norm = x_con_sq
        # max_val, _ = torch.max(x_con_sq, dim=1)
        # min_val, _ = torch.min(x_con_sq, dim=1)
        # x_norm = (x_con_sq - min_val) / (max_val - min_val)
        x_norm = x_norm.reshape(x_norm.shape[0], x_norm.shape[1] * x_norm.shape[2])
        vals, _ = x_norm.topk(int((1 - self.p) * x_norm.shape[1]), dim=1, sorted=False)
        threholds, _ = torch.min(vals, dim=1, keepdim=True)
        threholds = threholds.expand(threholds.shape[0], x.shape[2]*x.shape[3])
        mask = torch.where(x_norm - threholds < 0, torch.ones(x_norm.shape, device="cuda"),
                           torch.zeros(x_norm.shape, device="cuda"))
        mask = mask.reshape(mask.shape[0], x_con_sq.shape[1], x_con_sq.shape[2])
        x_iptc = self.sigmoid(x_con_sq)
        x_rand = torch.rand(size=x_iptc.shape).cuda()
        x_rand = torch.where(x_rand - 0.5 < 0, torch.ones(x_rand.shape, device="cuda"),
                             torch.zeros(x_rand.shape, device="cuda"))
        final_map = mask * x_rand * x_iptc + x_iptc * (1.0 - x_rand)
        final_map = final_map.unsqueeze(1)

        if self.training:
            return x * final_map
        else:
            return x # * self.sigmoid(x_con)


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
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)

        return out


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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.bn4 = nn.BatchNorm2d(512)
        self.drop = nn.Dropout(0.5)
        # self.fc5 = nn.Linear(512 * 8 * 8, 512)
        # self.bn5 = nn.BatchNorm1d(512)
        # self.fc6 = MyLoss(512, 509, s=64.0, m=0.55, easy_margin=True) # 659
        # self.fc6 = SoftMaxProduct(512, opt.num_classes)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, opt.num_classes)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.bn4(x)
        # x = x.view(x.size(0), -1)
        x = self.drop(x)
        feat = self.fc2(x)
        # x = self.fc5(x)
        # feat = self.bn5(x)
        # cal_score = self.fc6(feat)

        return feat, feat


def resnet_face18(use_se=False, pretrained=False, **kwargs):
    model = ResNetFace(IRBlock, [2, 3, 2, 2], use_se=use_se, **kwargs)
    if pretrained:
        # model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load("./output/res_new.pth"))
        # model.fc6 = MyLoss(512, opt.num_classes, s=64.0, m=0.3, easy_margin=True)
    return model
