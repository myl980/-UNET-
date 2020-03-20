# -*- coding: utf-8 -*-
'''将上采样写成反卷积的样式'''

import torch
import torch.nn as nn
from torch.nn import functional as F

#定义卷积层
class CNNLayer(nn.Module):
    def __init__(self, C_in, C_out):
        super(CNNLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C_in, C_out, 3, 1, 1),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(C_out, C_out, 3, 1, 1),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


#下采样用大步长卷积
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C, C, 3, 2, 1),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)

#上采样,使用反卷积
class UpSampling(nn.Module):
    def __init__(self,C):
        super().__init__()
        self.up_layer=nn.ConvTranspose2d(C,C//2,3,2,1,1)

    def forward(self, x,r):
        x=self.up_layer(x)

        return torch.cat((x, r), 1)

class MainNet(torch.nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        #这里用的灰度图，所以是1
        self.C1 = CNNLayer(1, 64)
        self.D1 = DownSampling(64)
        self.C2 = CNNLayer(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = CNNLayer(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = CNNLayer(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = CNNLayer(512, 1024)
        self.U1 = UpSampling(1024)
        self.C6 = CNNLayer(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = CNNLayer(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = CNNLayer(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = CNNLayer(128, 64)
        self.Th = torch.nn.Sigmoid()
        # 这里用的灰度图，输出和原图要对于，所以是1
        self.pre = torch.nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))
        return self.Th(self.pre(O4))


if __name__ == '__main__':
    a = torch.randn(2, 1, 256, 256).cuda()
    net = MainNet().cuda()
    print(net(a).shape)

