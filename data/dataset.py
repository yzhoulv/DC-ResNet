import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(3, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                # T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data_h = Image.open(img_path)
        data_h = data_h.convert('RGB')
        data_h = self.transforms(data_h)
        new_path = img_path.replace("Lock3d\\Lock3d", "Lock3d\\Lock3d\\GAN")
        data_l = Image.open(new_path)
        data_l = data_l.convert('RGB')
        data_l = self.transforms(data_l)
        label = np.int32(splits[1])
        return data_h.float(), data_l.float(), label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='E:\\paperData\\',
                      data_list_file='E:\\paperData\\BFtrain.txt',
                      phase='train',
                      input_shape=(4, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=32)
    for i, (data, label) in enumerate(trainloader):
        data = data[:, 0:3, :, :]
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
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
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)