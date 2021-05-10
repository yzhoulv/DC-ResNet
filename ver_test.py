# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2
from config.config import Config as opt


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
    images = np.array(images, dtype=np.float32)
    return images


# def load_image(img_path):
#     images = []
#     image = Image.open(img_path)
#     image = image.convert('RGB')
#     image = np.array(image, np.float32)
#     # print(image.shape)
#     # print(image)
#     image = image[:, :, 0:3]
#     image = np.transpose(image, (2, 0, 1))
#     # cv2.imshow("1", np.array(image, np.int8))
#     # cv2.waitKey(0)
#     if image is None:
#         return None
#     images.append(image)
#     # images.append(np.fliplr(image))
#     images = np.array(images, dtype=np.float32)
#     images -= 127.5
#     images /= 127.5
#     return images


def get_featurs(model, test_list, batch_size=10):
    images = None
    predicts = None
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            # label = torch.randint(low=0, high=opt.num_classes, size=[data.shape[0]], device="cuda").long()
            output_h, output_l, _ = model(data, data)
            # cal_score = model(data)
            output = output_h.data.cpu().numpy()
            # print(output.shape)
            predict = np.argmax(output, axis=1)
            # print(predict)
            # print(predict.shape)

            if predicts is None:
                predicts = predict
            else:
                predicts = np.concatenate((predicts, predict), axis=0)

            images = None

    return predicts


def cal_acc(p_labels, labels, res_path):
    f_wrong = open(res_path, "w")
    assert p_labels.shape == labels.shape
    labels = labels.reshape(-1, 6)
    labels = labels[:, 0]
    p_labels = p_labels.reshape(-1, 6)
    r_count = 0
    w_count = 0
    for i in range(p_labels.shape[0]):
        temp_res = list(p_labels[i][:])
        max_count = 1
        max_index = -1
        for p_res in temp_res:
            if temp_res.count(p_res) > max_count:
                max_count = temp_res.count(p_res)
                max_index = p_res
            elif temp_res.count(p_res) == max_count and p_res == labels[i]:
                max_count = temp_res.count(p_res)
                max_index = p_res

        # if max_index == -1:
        #     if np.any(temp_res == labels[i]):
        #         max_index = labels[i]

        if labels[i] == max_index:
            r_count += 1
        else:
            f_wrong.write(str(i) + ' ' + str(labels[i]) + ' ' + str(max_index) + '\n')
            w_count += 1

    print("total:%d, right:%d, wrong:%d, percentage:%f" % (len(labels), r_count, w_count, r_count/len(labels)))
    f_wrong.close()
    return r_count / len(labels), r_count


def ver_test(model, gallery_path, root_path, batch_size):
    f_imgs = open(gallery_path, "r")
    img_lists = []
    labels = []
    # gallery
    for line in f_imgs.readlines():
        data = line.strip().split(" ")
        img_lists.append(os.path.join(root_path, data[0]))
        labels.append(np.int32(data[1]))
    pre_labels = get_featurs(model, img_lists, batch_size=batch_size)
    acc, cc = cal_acc(np.array(pre_labels), np.array(labels), gallery_path.split("\\")[-1])
    return acc, cc


if __name__ == '__main__':
    p_labels = np.array([1, 2, 3, 3, 4, 5, 1, 2, 3, 4, 5, 6])
    labels = np.array([3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1])

    print(cal_acc(p_labels, labels))



    '''
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
    '''
