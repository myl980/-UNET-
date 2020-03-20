# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from Graduation_Project.Screws.U_Net import Attention_UNet
from Graduation_Project.Screws.U_Net import mydatasets

path = r'Datasets03'
module = r'models/attention_512_32_bce03_.pkl'
img_save_path = r'train_result'

net = Attention_UNet.MainNet(32).cuda()
optimizer = torch.optim.Adam(net.parameters())

#数据量较大，均方差效果一般；同时要学习的是轮廓信息和对应分类，所以用二分类效果要好点
loss_func = nn.BCELoss()

batch_size=2
dataloader = DataLoader(mydatasets.MKDataset(path), batch_size=batch_size, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
    print('module is loaded !')
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

batch=0
while True:
    batch += 1
    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.cuda()
        ys = ys.cuda()

        xs_ = net(xs)

        loss = loss_func(xs_, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print('epochs--{}--{}/{},input_count:{},loss:{}'.format(batch,i,len(dataloader),(i + 1) * batch_size, loss))

        torch.save(net.state_dict(), module)
        # print(i)
        # print('module is saved !')


        '''将输入图片、输出图片、标签图片展示在一张图上'''
        x = xs[0]
        x_ = xs_[0]
        y = ys[0]
        z = torch.cat((x, x_, y), 2)
        img_save = transforms.ToPILImage()(z.cpu())
        img_save.save(os.path.join(img_save_path, '{}.png'.format(i)))

    # 每20轮另外保存一个模型
    if batch % 5 == 0:
        torch.save(net.state_dict(), f'models/attention_512_32_bce03_{batch}.pkl')

