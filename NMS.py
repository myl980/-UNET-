# -*- coding: utf-8 -*-
# @Time: 2020-02-11 15:09
# Author: Trible
'''
获取角点坐标，
由于每个角点是由多个像素点构成，所以需要用nms的思想进行筛选
'''
import numpy as np
import cv2

#求两个点的欧式距离
def distance(a_dot, b_dots):
    dis = np.sqrt((a_dot[0] - b_dots[:, 0])**2 + (a_dot[1] - b_dots[:, 1])**2)
    return dis

#测试的图为黑白图(做图片是高斯核做的)，bri_thresh为像素值阈值，大于阈值的就是角点
#matrix就是图片(图片每个像素点的像素值)；
# dis_thresh是距离的阈值,若小于阈值，则他们是同一个角点的，只保留像素值最大的那个点
def nms(matrix, bri_thresh=150, dis_thresh=5):

    if matrix.shape[0] == 0:
        return np.array([])

    # print(matrix.shape)
    #找到角点在matrix中的索引(坐标，是HW形状)
    index = np.argwhere(matrix > bri_thresh)
    # print(index)

    #得到所有符合要求的像素点的像素值
    value = matrix[matrix > bri_thresh]

    #value.argsort()是将value排序
    index = index[value.argsort()]
    #保存最终符合条件的像素点的索引(坐标)
    index_list = []

    #用NMS方法的思想保留角点
    while index.shape[0] > 1:
        a_dot = index[0]
        b_dots = index[1:]
        index_list.append(a_dot)
        dis = distance(a_dot, b_dots)
        #大于dis_thresh保留下来
        index = b_dots[np.where(dis > dis_thresh)]
    if index.shape[0] > 0:
        index_list.append(index[0])

    return np.stack(index_list)

# 聚类螺钉
def bolt_nms(index, dis_thresh=50):
    bolt_list = []
    while index.shape[0] >= 3:
        a_dot = index[0]
        b_dots = index[0:]
        dis = distance(a_dot, b_dots)
        _index = b_dots[np.where(dis < dis_thresh)]
        #判断螺钉的角点数量必须大于3
        if _index.shape[0] >= 3:
            bolt_list.append(_index)
        index = b_dots[np.where(dis > dis_thresh)]
    return bolt_list

if __name__ == '__main__':
    #opencv打开图片的形状是HWC，所以索引的第一个值高(H)，第二个是宽(W)
    #cv2.IMREAD_GRAYSCALE表示以灰度图的形式打开图片
    #0004.png就是黑白图，是unet的输出图或标签图
    img = cv2.imread(r"Datasets\ann\0004.png", cv2.IMREAD_GRAYSCALE)
    index = nms(img)
    print(index.shape)
    #对应图中的形状是HW
    # print(index)

    xx=bolt_nms(index)
    print(xx)
