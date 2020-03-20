# -*- coding: utf-8 -*-
'''
数据集是正方形，可以直接resize为一样的大小即可
'''
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import os

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            ])

class MKDataset(Dataset):
    def __init__(self, path):
        self.path = path
        # 由于原图数量远多于标签图，而只使用有标签的图片，因此只能读取标签文件夹中的图名来进行训练
        self.name = os.listdir(os.path.join(path, 'ann'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):

        #原图路径
        img1_path = os.path.join(self.path, 'train_test01')
        #标签图路径
        img0_path = os.path.join(self.path, 'ann')

        name = self.name[index]
        #打开图片
        #原图，
        img1 = Image.open(os.path.join(img1_path,name))
        #标签图,
        img0 = Image.open(os.path.join(img0_path,name))

        #缩放图片大小
        img1_use = img1.resize(64)
        img0_use = img0.resize(64)

        #转为灰度图
        black1 = img1_use.convert('L')
        black0 = img0_use.convert('L')

        return transform(black1), transform(black0)


if __name__ == '__main__':
    i = 1
    dataset = MKDataset(r'E:\Datasets')
    for a, b in dataset:
        print(i)
        print(a)
        print(b)
        i+=1

