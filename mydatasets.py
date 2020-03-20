# -*- coding: utf-8 -*-
'''
由于数据集的图片大小不一，且宽和高不一样大小，
所以这里的做法是制作一个固定大小(这个是256*256)背景板，
将图片粘上去(效果与短边填充，然后resize是一样的)

如果数据集是正方形，可以直接resize为一样的大小即可
'''
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import os

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            ])

image_size=512
#将图片转为张量
black = torchvision.transforms.ToPILImage()(torch.zeros(3, image_size, image_size))


class MKDataset(Dataset):
    def __init__(self, path):
        self.path = path
        # 由于原图数量远多于标签图，而只使用有标签的图片，因此只能读取标签文件夹中的图名来进行训练
        self.name = os.listdir(os.path.join(path, 'ann'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        '''为了加快训练速度将图片缩放为image_size*image_size；注意：必须是等比例缩放'''

        #创建两个(原图和标签图)背景色为0的画板，大小为image_size*image_size，后面直接粘在这上面
        black1 = torchvision.transforms.ToPILImage()(torch.zeros(3, image_size, image_size))
        black0 = torchvision.transforms.ToPILImage()(torch.zeros(3, image_size, image_size))
        name = self.name[index]
        #标签图是png,而原图是JPG，所以将后三位换为JPG
        namejpg = name[:-3] + 'jpg'
        #原图
        img1_path = os.path.join(self.path, 'train_test01')
        #标签图
        img0_path = os.path.join(self.path, 'ann')
        #打开图片
        #原图，
        img1 = Image.open(os.path.join(img1_path, namejpg))
        #标签图,
        img0 = Image.open(os.path.join(img0_path, name))

        img1_size = torch.Tensor(img1.size)  # WH
        #较大边
        l_max_index = img1_size.argmax()
        #image_size与较大边的比例
        ratio = float(image_size) / img1_size[l_max_index.item()]
        #计算resize为image_size后的W和H的长度
        img1_re2size = img1_size * ratio

        img1_use = img1.resize(img1_re2size)
        img0_use = img0.resize(img1_re2size)

        w, h = img1_re2size.tolist()
        #直接将图粘在创建好的全0画板上，相当于用黑色来填充了较短边
        black1.paste(img1_use, (0, 0, int(w), int(h)))
        black0.paste(img0_use, (0, 0, int(w), int(h)))

        #转为灰度图
        black1 = black1.convert('L')
        black0 = black0.convert('L')

        return transform(black1), transform(black0)


if __name__ == '__main__':
    i = 1
    dataset = MKDataset(r'E:\Datasets')
    for a, b in dataset:
        print(i)
        print(a)
        print(b)
        i+=1

