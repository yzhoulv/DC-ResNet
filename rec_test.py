# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import torch
import numpy as np
from config.config import Config as opt
from PIL import Image
import os
import cv2
import torchvision


def load_image(img):
    images = []
    img = cv2.imread(img, 1)
    img = np.float32(cv2.resize(img, (128, 128))) / 255
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    # means = [0.5, 0.5, 0.5]
    # stds = [0.5, 0.5, 0.5]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    images.append(preprocessed_img)
    images.append(np.fliplr(preprocessed_img))
    images = np.array(images, dtype=np.float32)
    return images


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt = i
            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            # label = torch.randint(low=0, high=opt.num_classes, size=[data.shape[0]], device="cuda").long()
            feat_h, feat_l, _ = model(data, data)
            output = feat_h.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)
            # feature = feat_h.data.cpu().numpy()
            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def cal_acc(gal, prb, res_path):
    # f_wrong = open(res_path, "w")
    gal_data = np.array(gal[0]).reshape(-1, 512*2)
    gal_label = np.array(gal[1]).reshape(-1, 1)
    # print(gal_data.shape)
    prb_data = np.array(prb[0]).reshape(-1, 512*2)
    prb_label = np.array(prb[1]).reshape(-1, 1)
    # print(prb_data.shape)
    res_mal = np.matmul(prb_data, gal_data.T)
    gal_norm = np.linalg.norm(gal_data, axis=1).reshape(-1, 1)
    prb_norm = np.linalg.norm(prb_data, axis=1).reshape(-1, 1)
    acc_arr = res_mal / (np.tile(prb_norm, (1, gal_norm.shape[0])) * gal_norm.T)
    acc_index = np.argmax(acc_arr, axis=1)  # 每行最小值索引
    # print(acc_index.shape)
    w_count = 0
    r_count = 0
    for i in range(prb_data.shape[0]):
        if prb_label[i] == gal_label[acc_index[i]]:
            r_count += 1
        else:
            w_count += 1
            # f_wrong.write(str(i) + ' ' + str(prb_label[i]) + ' ' + str(gal_label[acc_index[i]]) + '\n')

    print("total:%d, right:%d, wrong:%d" % (len(prb_label), r_count, w_count))
    # f_wrong.close()
    return r_count/len(prb_label), w_count/len(prb_label)


def cal_acc_c(gal, prb, res_path):
    # f_wrong = open(res_path, "w")
    gal_data = np.array(gal[0]).reshape(-1, 512)
    gal_label = np.array(gal[1]).reshape(-1, 1)
    # print(gal_data.shape)
    prb_data = np.array(prb[0]).reshape(-1, 512)
    prb_label = np.array(prb[1]).reshape(-1, 1)
    # print(prb_data.shape)
    w_count = 0
    r_count = 0
    for i in range(prb_data.shape[0]):
        feat = prb_data[i]
        feats = np.tile(feat, [gal_data.shape[0], 1])
        feats = np.linalg.norm(feats-gal_data, axis=1).reshape(1, -1)
        min_index = np.argmin(feats)
        if gal_label[min_index] == prb_label[i]:
            r_count += 1
        else:
            w_count += 1
    # res_mal = np.matmul(prb_data, gal_data.T)
    # gal_norm = np.linalg.norm(gal_data, axis=1).reshape(-1, 1)
    # prb_norm = np.linalg.norm(prb_data, axis=1).reshape(-1, 1)
    # acc_arr = res_mal / (np.tile(prb_norm, (1, gal_norm.shape[0])) * gal_norm.T)
    # acc_index = np.argmax(acc_arr, axis=1)  # 每行最小值索引
    # # print(acc_index.shape)
    # w_count = 0
    # r_count = 0
    # for i in range(prb_data.shape[0]):
    #     if prb_label[i] == gal_label[acc_index[i]]:
    #         r_count += 1
    #     else:
    #         w_count += 1
    #         # f_wrong.write(str(i) + ' ' + str(prb_label[i]) + ' ' + str(gal_label[acc_index[i]]) + '\n')

    print("total:%d, right:%d, wrong:%d" % (len(prb_label), r_count, w_count))
    # f_wrong.close()
    return r_count/len(prb_label), w_count/len(prb_label)


def rec_test(model, gallery_path, probe_path, root_path, batch_size):
    gallery = [[], []]
    probe = [[], []]
    f_gallery = open(gallery_path, "r")
    f_probe = open(probe_path, "r")
    img_lists = []
    # gallery
    for line in f_gallery.readlines():
        data = line.strip().split(" ")
        img_lists.append(os.path.join(root_path, data[0]))
        gallery[1].append(np.int32(data[1]))
    features, cnt = get_featurs(model, img_lists, batch_size=batch_size)
    gallery[0].extend(features)
    # probe
    img_lists = []
    for line in f_probe.readlines():
        data = line.strip().split(" ")
        img_lists.append(os.path.join(root_path, data[0]))
        probe[1].append(np.int32(data[1]))
    features, cnt = get_featurs(model, img_lists, batch_size=batch_size)
    probe[0].extend(features)
    acc, _ = cal_acc(gallery, probe, probe_path.split("\\")[-1])
    return acc, cnt


if __name__ == '__main__':

    imgs = load_image("E:\\paperData\\swjtu\\Test\\0011\\0011_001_XYYH_00_00_00_64_75_15.bmp")
    # img = img[:, 0:3, :, :]
    # imgs, labels = data
    # print imgs.numpy().shape
    # print data.cpu().numpy()
    # if i == 0:
    # img = torchvision.utils.make_grid(img).numpy()
    img = imgs[0][0:3, :, :]
    print(img.shape)
    # print label.shape
    # chw -> hwc
    img = np.transpose(img, (1, 2, 0))
    # img *= np.array([0.229, 0.224, 0.225])
    # img += np.array([0.485, 0.456, 0.406])
    img += np.array([1, 1, 1])
    img *= 127.5
    img = img.astype(np.uint8)
    img = img[:, :, [2, 1, 0]]

    cv2.imshow('img', img)
    cv2.waitKey(0)