'''
以每个点为中心做一个高斯分布，做成黑白的二值图，
将MTCNN关键点标准方式转为UNET的关键点标注方式
即：将关键点由单个像素点转为多个像素点，且整体为黑白图
'''
import os
import cv2 as cv
import numpy as np
from xml.etree import ElementTree
import matplotlib.pyplot as plt

# 读取xml数据，可以改成json,输出(n, 6, 2),n是螺钉个数，6是6个角点，2是每个角点的x,y
def read_data(path):
    tree = ElementTree.parse(path)
    root = tree.getroot()
    file_name = root.find("path").text
    object = root.find("outputs").find("object")
    list1 = []
    for item in object:
        polygon = item.find("polygon")
        for i, child in enumerate(polygon):
            list1.append(int(child.text))
    arr = np.array(list1).reshape([-1, 6, 2])
    return arr

# 生成高斯核，img_width, img_heigh图片宽高，c_x, c_y高斯中心点，sigma高斯核大小
def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap

img_size=1024
# 遍历xml文件，生成背景图， 大小为img_size*img_size
for xml_name in os.listdir(r"Datasets\outputs_xml"):
    img = np.zeros((img_size, img_size, 1))

    arr = read_data(os.path.join(r"Datasets\outputs_xml",xml_name))
    for data in arr:
        for c in data:
            #1808是原图片大小，这里要按比例构建
            heatmap = CenterLabelHeatMap(img_size, img_size, int(c[0]*img_size/1808), int(c[1]*img_size/1808), 3)[:, :, np.newaxis]
            img += heatmap
    img = img * 255
    cv.imwrite(f"{xml_name.split('.')[0]}.png", img.astype(np.uint8))
    print(f"{xml_name.split('.')[0]}图片完成")