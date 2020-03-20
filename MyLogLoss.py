# -*- coding: utf-8 -*-
# @Time: 2020-02-12 10:03
# Author: Trible
'''
Log loss
'''
import torch

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, reduction='elementwise_mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, _input, target):
        pt = _input
        y=target
        loss = -(y*torch.log(pt)+(1-y)*torch.log(1-pt))
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
