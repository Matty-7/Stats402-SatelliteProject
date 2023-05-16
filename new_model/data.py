from __future__ import print_function
import argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR
import os

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, transform=None, target_transform=None, prefix=None):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.fname_list = None
        self.prefix = prefix
        self.base_dir = os.path.join(prefix, base_dir)
        self.target_transform = target_transform
        self.create_fname_list()
        self.img_labels = self.fname_list
        

    def create_fname_list(self):
        # self.fname_list = os.listdir(os.path.join(self.base_dir, '2012/label'))
        # self.fname_list = list(filter(lambda x: x.find("jpg") != -1, self.fname_list))
        # self.fname_list = list(map(lambda x: x[5: ], self.fname_list))
        raw = os.listdir(self.base_dir + '/label')
        # print(len(raw))
        # print(self.base_dir + '/label')
        self.fname_list = list(filter(lambda x: os.stat(os.path.join(self.prefix, 'data/resized/label/') + x)[6] > 3000, raw))
        self.fname_list = list(filter(lambda x: x.find("jpg") != -1, self.fname_list))
        self.fname_list = list(map(lambda x: x[13: ], self.fname_list))
        # print(len(self.fname_list))


    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # TODO: refactor
        img_path = os.path.join(self.base_dir, '2012/label', '2012_label_' + self.fname_list[idx])
        img_path2 = os.path.join(self.base_dir, '2016/label', '2016_label_' + self.fname_list[idx])
        image = plt.imread(img_path)
        image2 = plt.imread(img_path2)
        p2 = np.array(np.zeros((3, 256, 256)))
        p2[0, ...] = image[..., 0]
        p2[1, ...] = image[..., 1]
        p2[2, ...] = image[..., 2]
        p3 = np.array(np.zeros((3, 256, 256)))
        p3[0, ...] = image2[..., 0]
        p3[1, ...] = image2[..., 1]
        p3[2, ...] = image2[..., 2]
        label_path = os.path.join(self.base_dir, 'label', 'change_label_' + self.fname_list[idx])
        # print(label_path)
        image_label = plt.imread(label_path)
        label = np.divide(image_label[..., 0], 255).astype(int)
        # lab = np.array(np.zeros((1, 256, 256)))
        # lab[0, :, :] = label
        tor = torch.from_numpy(p2)
        tor2 = torch.from_numpy(p3)
        feature = self.normalize_image(tor)
        feature2 = self.normalize_image(tor2)
        # print(img_path)

        # return feature, feature2, torch.from_numpy(label).float()
        return {'I1': feature, 'I2': feature2, 'label': torch.from_numpy(label).float()}