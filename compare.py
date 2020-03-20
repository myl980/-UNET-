# -*- coding: utf-8 -*-
'''
对比两张图片的角点坐标
'''
import numpy as np
import cv2
from Graduation_Project.Screws.U_Net.NMS import nms

'''
由于NMS后的螺钉顺序和螺钉的角点顺序都是混乱的，
大概率与标签图的顺序不同，所以不能直接对比
(这里直接对比是正确的是因为两张图相同)
'''

#img1是标签图，img2是输出图
def compared(img1,img2):
    index1=nms(img1)
    index2=nms(img2)

    if index1.shape ==index2.shape:
        #判断预测角点与标签图角点坐标是否一致
        result = np.all(index1 == index2, axis=1)

        #计算预测角点与标签图角点的偏移量
        # result=index1-index2
        # print(np.sum(result,axis=1))

    else:
        result='螺丝数量都预测错了'

    return result

if __name__ == '__main__':

    img1=cv2.imread(r"Datasets\result\target_img\1.png", cv2.IMREAD_GRAYSCALE)
    img2=cv2.imread(r"Datasets\result\target_img\1.png", cv2.IMREAD_GRAYSCALE)

    result=compared(img1,img2)
    print(result)


