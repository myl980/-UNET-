# -*- coding: utf-8 -*-
'''
对比两张图片的角点坐标
'''
import numpy as np
import cv2
from Graduation_Project.Screws.U_Net.NMS import nms,bolt_nms

'''
以螺钉中心点的欧氏距离来确定对应的螺钉
再用对应螺钉每个角点的欧式距离确定对应角点

注：还没开始写
'''

def compared(img1,img2,threshed1=100,threshed2=5):
    index1 = nms(img1)
    index2 = nms(img2)

    #判断两张图角点数量是否相等
    if index1.shape==index2.shape:
        #聚类螺钉的角点
        bolt_index_1=bolt_nms(index1)
        # print('bolt_index_1:',bolt_index_1)
        bolt_index_2=bolt_nms(index2)
        # print('bolt_index_2:', bolt_index_2)

        '''
        确定对应螺钉：用螺钉的中心点距离来确定
        距离最近的就是对应的螺钉
        '''
        bolt_new_1=[]
        bolt_new_2=[]
        for bolt1 in bolt_index_1:
            pass

        # '''
        # 确定对应的角点：也是用欧式距离
        # '''
        # bolt_final_1 = []
        # bolt_final_2 = []
        # for i in range(len(bolt_new_1)):
        #     bolt1=[]
        #     bolt2=[]
        #     for corner1 in bolt_new_1[i]:
        #         for corner2 in bolt_new_2[i]:
        #             # print(corner1,corner2)
        #             dis=np.linalg.norm(corner1 - corner2)
        #             # print('dis:',dis)
        #
        #             if dis < 5:
        #                 bolt1.append(corner1.tolist())
        #                 bolt2.append(corner2.tolist())
        #
        #
        #     bolt_final_1.append(np.array(bolt1))
        #     bolt_final_2.append(np.array(bolt2))
        #
        # # print(bolt_final_1)
        # # print(bolt_final_2)
        #
        # result=[]
        # for i in range(len(bolt_final_1)):
        #     dises=np.linalg.norm(bolt_final_1[i] - bolt_final_2[i],axis=1)
        #     # print(dises)
        #     compare=dises<threshed2
        #     result.append(compare.tolist())
        # # print(result)

    else:
        print('两张图角点数都不同')

    return result



if __name__ == '__main__':
    #
    # img1=cv2.imread(r"E:\Datasets\ann\0005.png", cv2.IMREAD_GRAYSCALE)
    # # print(img1.shape)
    # img2=cv2.imread(r"E:\Datasets\ann\0005.png", cv2.IMREAD_GRAYSCALE)

    #这张图会报错，因为有一个螺钉的角点数是5，而其他为6，不能转为array数据类型
    img1=cv2.imread(r"002.png", cv2.IMREAD_GRAYSCALE)
    img2=cv2.imread(r"002.png", cv2.IMREAD_GRAYSCALE)

    result=compared(img1,img2)
    print(result)



