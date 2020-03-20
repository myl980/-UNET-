# -*- coding: utf-8 -*-
# @Time: 2020-02-12 10:03
# Author: Trible
'''
非标准focal loss
没有gamma参数，alfa为0.75
'''
import torch

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    #_input是网络的输出，target是标签
    def forward(self, _input, target):
        pt = _input
        alpha = self.alpha
        loss = - (alpha *target* ((1 - pt) ** self.gamma) * torch.log(pt) - \
               (1 - alpha) * (1 - target) * (pt ** self.gamma)  * torch.log(1 - pt))
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

if __name__ == "__main__":
    focal_loss = BCEFocalLoss()
    x = torch.nn.Sigmoid()(torch.randn((3, 3,3,3)))
    y = torch.nn.Sigmoid()(torch.randn((3, 3,3,3)))
    loss = focal_loss(x, y)
    print(loss)
