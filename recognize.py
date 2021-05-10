from __future__ import print_function
from data.dataset import Dataset
import torch
from config.config import Config
import numpy as np
from models.resnet import resnet_face18


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_acc(gal, prb):
    print("calculate result!")
    gal_data = np.array(gal[0]).reshape(-1, 512)
    gal_label = np.array(gal[1]).reshape(-1, 1)
    print(gal_data.shape)
    prb_data = np.array(prb[0]).reshape(-1, 512)
    prb_label = np.array(prb[1]).reshape(-1, 1)
    print(prb_data.shape)
    res_mal = np.matmul(prb_data, gal_data.T)
    gal_norm = np.linalg.norm(gal_data, axis=1).reshape(-1, 1)
    prb_norm = np.linalg.norm(prb_data, axis=1).reshape(-1, 1)
    acc_arr = res_mal / (np.tile(prb_norm, (1, gal_norm.shape[0])) * gal_norm.T)
    acc_index = np.argmax(acc_arr, axis=1)  # 每行最小值索引
    print(acc_index.shape)
    w_count = 0
    r_count = 0
    for i in range(prb_data.shape[0]):
        if prb_label[i] == gal_label[acc_index[i]]:
            r_count += 1
        else:
            w_count += 1
            # print("prb_label:%d, gal_label:%d" % (prb_label[i], gal_label[acc_index[i]]))

    print("total:%d, right:%d, wrong:%d" % (len(prb_label), r_count, w_count))

    return r_count/len(prb_label), w_count/len(prb_label)


if __name__ == '__main__':
    opt = Config()
    device = torch.device("cuda")

    # gallery
    gallery = [[], []]
    prob = [[], []]
    test_dataset = Dataset(opt.test_root, opt.test_list_gal, phase='test', input_shape=opt.input_shape)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                  batch_size=opt.test_batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers)

    print('gallery {} train iters per epoch:'.format(len(testloader)))
    # model = Led3DNet(pretrained=True)
    # model = MyNet(pretrained=True)
    model = resnet_face18(use_se=True, pretrained=True)
    model.to(device)
    model.eval()
    for ii, data in enumerate(testloader):
        data_input, label = data
        data_input = data_input.to(device)
        feat, cls_score = model(data_input)
        gallery[0].extend(feat.data.cpu().numpy())
        gallery[1].extend(label)

    # prob
    test_dataset = Dataset(opt.test_root, opt.test_list_prob, phase='test', input_shape=opt.input_shape)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                  batch_size=opt.test_batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers)

    print('prob {} train iters per epoch:'.format(len(testloader)))
    for ii, data in enumerate(testloader):
        data_input, label = data
        data_input = data_input.to(device)
        feat, cls_score = model(data_input)
        prob[0].extend(feat.data.cpu().numpy())
        prob[1].extend(label)

    print(gallery[0][0].shape)
    f = open("res.txt", "a")
    f.write(str(gallery[0][0][0]) + "\n")
    print(gallery[1][0])

    print(cal_acc(gallery, prob))

