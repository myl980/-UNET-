# -*- coding: utf-8 -*-
'''
测试和训练代码几乎一样，使用的也是训练时的图片,批量进行测试
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from Graduation_Project.Screws.U_Net import U_Net01
from Graduation_Project.Screws.U_Net import mydatasets

path = r'Datasets\test__'
module = r'models/module.pkl'
img_save_path = r'result\compare'
out_img_path=r'result\out_img'
target_img_path=r'result\target_img'
batch = 1

net = U_Net01.MainNet().cuda()
dataloader = DataLoader(mydatasets.MKDataset(path), batch_size=4, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
    print('module is loaded !')

for path in [img_save_path,out_img_path,target_img_path]:
    if not os.path.exists(path):
        os.mkdir(path)



for i, (xs, ys) in enumerate(dataloader):
    xs = xs.cuda()
    ys = ys.cuda()

    xs_ = net(xs)

    '''将输入图片、输出图片、标签图片展示在一张图上'''
    x = xs[0]
    x_ = xs_[0]
    y = ys[0]
    z = torch.cat((x, x_, y), 2)

    # 对比效果图
    img_save = transforms.ToPILImage()(z.cpu())
    # 输出图
    out_img = transforms.ToPILImage()(x_.cpu())
    # 标签图
    target_img = transforms.ToPILImage()(y.cpu())

    img_save.save(os.path.join(img_save_path, '{}.png'.format(batch)))
    out_img.save(os.path.join(img_save_path, '{}.png'.format(batch)))
    target_img.save(os.path.join(img_save_path, '{}.png'.format(batch)))
    batch += 1


