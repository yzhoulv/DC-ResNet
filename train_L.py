from __future__ import print_function
import os
from data.dataset import Dataset
from torch.utils import data
from models.metrics import *
from utilss.visualizer import Visualizer
import torch
import numpy as np
import time
from config.config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from models.focal_loss import FocalLoss
from models.resnet import resnet_face18
from rec_test import rec_test

max_acc = 0

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + iter_cnt + '.pth')
    # save_name = os.path.join(save_path, name + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def test(model):
    model.eval()
    acc_nu, nu_num = rec_test(model, opt.test_gallery, opt.probe_nu, opt.test_root, opt.test_batch_size)
    acc_oc, oc_num = rec_test(model, opt.test_gallery, opt.probe_oc, opt.test_root, opt.test_batch_size)
    acc_fe, fe_num = rec_test(model, opt.test_gallery, opt.probe_fe, opt.test_root, opt.test_batch_size)
    acc_ps, ps_num = rec_test(model, opt.test_gallery, opt.probe_ps, opt.test_root, opt.test_batch_size)
    acc_tm, tm_num = rec_test(model, opt.test_gallery, opt.probe_tm, opt.test_root, opt.test_batch_size)
    acc_total = (acc_nu * nu_num + acc_ps * ps_num + acc_oc * oc_num + acc_fe * fe_num + acc_tm * tm_num) / (
                nu_num + ps_num + oc_num + fe_num + tm_num)
    print("epoch:{}, total:{}, acc_fe:{}, acc_nu:{}, acc_oc:{}, acc_ps:{}, acc_tm:{}, ".format(i, acc_total, acc_fe,
                                                                                               acc_nu,
                                                                                               acc_oc,
                                                                                               acc_ps,
                                                                                               acc_tm) + "\n")

    global max_acc
    if acc_total > max_acc:
        max_acc = acc_total
        save_model(model, opt.checkpoints_path, opt.backbone, "best")
    time_str = time.asctime(time.localtime(time.time()))
    print("time:{}, epoch:{}, test acc:{}, best acc:{}".format(time_str, i, acc_total, max_acc))


if __name__ == '__main__':

    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")
    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # model = MyNet(pretrained=True)
    model = resnet_face18(pretrained=True)
    # criterion = CircleLoss(m=0.25, gamma=80)

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.0)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.3, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    elif opt.metric == 'my_loss':
        metric_fc = MyLoss(opt.num_classes, opt.num_classes, s=30, m=0.3, easy_margin=opt.easy_margin)
    else:
        metric_fc = SoftMaxProduct(512, opt.num_classes)
        # metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    # print(model)
    model.to(device)
    metric_fc.to(device)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    for i in range(opt.max_epoch):
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feats, cls_score = model(data_input)
            # cls_score = nn.functional.normalize(cls_score)
            # print(feature.data.cpu().numpy())
            # output = F.softmax(feature, dim=0)
            # output, weight_loss = metric_fc(cls_score, label)
            output = cls_score
            loss_infer = criterion(cls_score, label)
            loss = loss_infer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s infer loss:{} acc:{}'.format(time_str, i, ii, speed,
                                                                                         loss_infer.item(), acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()
                # save_model(model, opt.checkpoints_path, opt.backbone, "new")

        test(model)
        save_model(model, opt.checkpoints_path, opt.backbone, "new")
        scheduler.step()

        # if opt.display:
        #     visualizer.display_current_results(iters, acc, name='test_acc')
