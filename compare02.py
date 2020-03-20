# -*- coding: utf-8 -*-
'''
对比两张图片的角点坐标
'''
import numpy as np
import cv2
from Graduation_Project.Screws.U_Net.NMS import nms,bolt_nms


'''
由于NMS后的螺钉顺序和螺钉的角点顺序都是混乱的，
大概率与标签图的顺序不同，所以不能直接对比，需要先进行排序，找到对应的螺钉和角点

由于左下角螺钉坐标和右上角螺钉坐标总值是很接近的,所以没法按总值排序来区分螺钉
所以这段代码是错的！！！
而且如果螺钉角点数量不同会报错，因为这样无法转为arrary数据类型，内部维度不同
'''

#threshed1是螺钉角点总偏移量，threshed2是单个角点偏移量
def compared(img1,img2,threshed1=25,threshed2=5):
    index1 = nms(img1)
    index2 = nms(img2)

    #判断两张图角点数量是否相等
    if index1.shape==index2.shape:
        #聚类螺钉的角点
        bolt_index_1=bolt_nms(index1)
        # print(bolt_index1)
        bolt_index_2=bolt_nms(index2)

        '''
        排螺钉的顺序，并判断螺钉是否预测准确
        原理：由于两张图大小一致，螺钉各个角点坐标最接近的代表是同一个螺钉
        '''
        #图上每个螺丝角点坐标总值
        image1_bolt_value=[np.sum(i) for i in bolt_index_1]
        #获取按螺丝角点坐标总值排序的索引
        image1_bolt_index=np.array(image1_bolt_value).argsort()
        # print(image1_bolt_index)
        #按螺丝角点坐标总值排序
        image1_bolt_value.sort()
        # print(image1_bolt_value)

        # 图上每个螺丝角点坐标总值
        image2_bolt_value = [np.sum(i) for i in bolt_index_2]
        # 获取按螺丝角点坐标总值排序的索引
        image2_bolt_index = np.array(image2_bolt_value).argsort()
        # print(image2_bolt_index)
        # 按螺丝角点坐标总值排序
        image2_bolt_value.sort()
        # print(image2_bolt_value)

        #根据螺丝角点坐标总值排序的索引排序bolt_index
        bolt_index1=np.array(bolt_index_1)[image1_bolt_index]
        print('bolt_index1:',bolt_index1)
        bolt_index2=np.array(bolt_index_2)[image2_bolt_index]

        #确定每个螺丝角点总偏移量
        diff=np.array(image1_bolt_value)-np.array(image2_bolt_value)
        # print('diff:',diff.tolist())
        # #np.where(diff<10)是个元组，第一个才是numpy数据
        # diff_index=np.where(diff<10)[0]
        # print('type(diff_index):',type(diff_index))
        # print('diff_index:',diff_index)

        '''
        由于左下角螺钉坐标和右上角螺钉坐标总值是很接近的,所以没法按总值排序来区分螺钉
        '''
        # result=[]
        # for i in diff.tolist():
        #     #小于threshed1则此螺丝没有松动
        #     if i < threshed1:
        #         result.append(True)
        #     else:
        #         result.append({False})
        #浓缩上面的循环，diff.tolist()是array转列表
        result=[True if i<threshed1 else False for i in diff.tolist()]
        # print('result:',result)


        '''
        排角点顺序,并判断角点是否预测准确
        原理：角点坐标最相近的就是同一个角点
        即：x1-x2 + y1-y2 最小就指代一个角点，x1-x2 + y1-y2=x1+y1 - x2+y2
        索引x+y最相近的就是对应的角点
        '''
        #求出每个角点坐标的总值
        bolts_value1=np.sum(bolt_index1,axis=2)
        print(bolts_value1)
        bolts_value2 = np.sum(bolt_index2, axis=2)
        # print(bolts_value2)

        #按每个角点坐标的总值排序
        bolts_value1.sort(axis=1)
        # print(bolts_value1)
        bolts_value2.sort(axis=1)
        # print(bolts_value2)

        #求每个角点的偏移
        diff_corner=bolts_value1-bolts_value2
        # print(diff_corner)

        result02=[]
        for i in range(len(diff_corner.tolist())):
             result02.append([True if j<threshed2 else False for j in diff_corner.tolist()[i]])
        # print(result02)
    return result,result02

if __name__ == '__main__':
    #
    img1=cv2.imread(r"Datasets\result\target_img\1.png", cv2.IMREAD_GRAYSCALE)
    # print(img1.shape)
    img2=cv2.imread(r"Datasets\result\target_img\1.png", cv2.IMREAD_GRAYSCALE)

    # #这张图会报错，因为有一个螺钉的角点数是5，而其他为6，不能转为array数据类型
    # img1=cv2.imread(r"E:\train_data2161\002.png", cv2.IMREAD_GRAYSCALE)
    # img2=cv2.imread(r"E:\train_data2161\002.png", cv2.IMREAD_GRAYSCALE)

    result,result02=compared(img1,img2)
    print(result)
    print(result02)


