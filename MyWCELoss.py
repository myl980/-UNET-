# -*- coding: utf-8 -*-
# @Time: 2020-02-12 10:03
# Author: Trible
'''
加权交叉熵损失，WCELoss
交叉熵损失分别计算每个像素的交叉熵，然后对所有像素进行平均，这意味着我们默认每类像素对损失的贡献相等。
如果各类像素在图像中的数量不平衡，则可能出现问题，因为数量最多的类别会对损失函数影响最大，从而主导训练过程。
Long等提出了为每个类加权的交叉熵损失（WCE），以抵消数据集中存在的类不平衡。
'''
import torch

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, beta=0.5,reduction='elementwise_mean'):
        super().__init__()
        self.beta=beta
        self.reduction = reduction

    def forward(self, _input, target):
        pt = _input
        y=target
        loss = -(self.beta * y*torch.log(pt)+(1-y)*torch.log(1-pt))
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
