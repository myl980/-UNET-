'''
以每个点为中心做一个高斯分布，做成黑白的二值图，
将MTCNN关键点标准方式转为UNET的关键点标注方式
即：将关键点由单个像素点转为多个像素点，且整体为黑白图

注：这是直接制作图片，将关键点为白，其余为黑，(如果没有关键点，就是全黑)
这样生成的unet标签数据仅适用于单类别关键点检测(即不对关键点进行分类的任务)
'''
import os
import cv2 as cv
import numpy as np
from xml.etree import ElementTree
import json
import matplotlib.pyplot as plt

# 读取json数据,输出(n, 6, 2),n是螺钉个数，6是6个角点，2是每个角点的x,y
'''如果标的角点数不等于6，这里就会报错'''
def read_data(path):
    file = open(path, encoding='utf-8')
    data = json.load(file)
    img_path=data['path']
    list1 = []
    data = data['outputs']['object']
    for j in data:
        x1 = int(j['polygon']['x1'])
        y1 = int(j['polygon']['y1'])
        x2 = int(j['polygon']['x2'])
        y2 = int(j['polygon']['y2'])
        x3 = int(j['polygon']['x3'])
        y3 = int(j['polygon']['y3'])
        x4 = int(j['polygon']['x4'])
        y4 = int(j['polygon']['y4'])
        x5 = int(j['polygon']['x5'])
        y5 = int(j['polygon']['y5'])
        x6 = int(j['polygon']['x6'])
        y6 = int(j['polygon']['y6'])
        list1.extend([x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6])
    arr = np.array(list1).reshape([-1, 6, 2])
    return arr

# 生成高斯核，img_width, img_heigh图片宽高，c_x, c_y高斯中心点，sigma高斯核大小
def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    #生成坐标值
    #生成宽的坐标，1表示开始为值1，第一个img_width表示停止值为img_width，第二个img_width表示数量为img_width
    #np.linspace(start,stop,num)生成范围内一定数量均匀的数据
    X1 = np.linspace(start=1, stop=img_width, num=img_width)
    #生成高的坐标
    Y1 = np.linspace(1, img_height, img_height)
    #生成坐标矩阵
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    # print(X)
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    # print(heatmap)
    return heatmap

#注意生成图片大小要与高斯核大小成正比，若高斯核过大，图片过小，会生成一大坨，而不是关键点
img_size=1024
# 遍历xml文件，生成背景图， 大小为img_size*img_size
#E:\outputs存放的标签文件，如经历标注助手导出的json/xml等
for json_name in os.listdir(r"Datasets\outputs_json"):
    img = np.zeros((img_size, img_size, 1))
    arr = read_data(os.path.join(r"Datasets\outputs_json",json_name))
    for data in arr:
        #遍列每一个螺丝
        for c in data:
            # 1808是原图片大小，这里要按比例构建
            heatmap = CenterLabelHeatMap(img_size, img_size, int(c[0]*img_size/1808), int(c[1]*img_size/1808), 3)[:, :, np.newaxis]
            img += heatmap
    img = img * 255
    #保存在当前文件夹下，需要手动移动到标签数据集文件夹
    cv.imwrite(f"{json_name.split('.')[0]}.png", img.astype(np.uint8))
    print(f"{json_name.split('.')[0]}图片完成")