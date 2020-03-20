# -*- coding: utf-8 -*-
'''
对比两张图片的角点坐标
'''
import numpy as np
import cv2
from Graduation_Project.Screws.U_Net.NMS import nms,bolt_nms

'''
以每个螺钉角点的欧氏距离总值来确定对应的螺钉
再以对应螺钉每个角点的欧式距离确定对应角点
'''

def compared(img1,img2,threshed1=200,threshed2=5):
    index1 = nms(img1)
    # print('index1.shape:',index1.shape)
    index2 = nms(img2)
    # print('index2.shape:',index2.shape)

    #保存结果的列表
    result=[]
    #判断两张图角点数量是否相等
    if index1.shape==index2.shape:
        #聚类螺钉的角点
        bolt_index_1=bolt_nms(index1)
        # print('bolt_index_1:',bolt_index_1)
        # print('len(bolt_index_1):',len(bolt_index_1))
        bolt_index_2=bolt_nms(index2)
        # print('bolt_index_2:', bolt_index_2)

        '''
        确定对应螺钉：用欧式距离来求
        就算角点顺序不同，对应螺钉的欧式距离肯定要小于非对应螺钉的
        由于角点顺序不同，所以不能用这个来却螺丝是否松动
        '''
        bolt_new_1=[]
        bolt_new_2=[]
        for bolt1 in bolt_index_1:
            # print('bolt1:',bolt1)
            # np.linalg.norm(bolt- bolt2,axis=1)求螺钉所有角点各自的欧式距离
            # dis=[np.linalg.norm(bolt- bolt2) if bolt.shape==bolt2.shape else print('维度不同') for bolt2 in bolt_index_2]
            for bolt2 in bolt_index_2:
                if bolt1.shape==bolt2.shape:
                    # print('np.sum(np.linalg.norm(bolt1- bolt2,axis=1)):',np.sum(np.linalg.norm(bolt1- bolt2,axis=1)))
                    if np.sum(np.linalg.norm(bolt1- bolt2,axis=1))<threshed1:
                        bolt_new_1.append(bolt1)
                        bolt_new_2.append(bolt2)

        print('bolt_new_1:',bolt_new_1)
        print('bolt_new_2:',bolt_new_2)

        # print('len(bolt_new_1):',len(bolt_new_1))

        '''
        确定对应的角点：也是用欧式距离
        '''
        bolt_final_1 = []
        bolt_final_2 = []
        for i in range(len(bolt_new_1)):
            bolt1=[]
            bolt2=[]
            for corner1 in bolt_new_1[i]:
                for corner2 in bolt_new_2[i]:
                    # print(corner1,corner2)
                    dis=np.linalg.norm(corner1 - corner2)
                    # print('dis:',dis)

                    if dis < 5:
                        bolt1.append(corner1.tolist())
                        bolt2.append(corner2.tolist())


            bolt_final_1.append(np.array(bolt1))
            bolt_final_2.append(np.array(bolt2))

        # print('bolt_final_1:',bolt_final_1)
        # print('bolt_final_2:',bolt_final_2)


        # print('len(bolt_final_1):',len(bolt_final_1))
        for i in range(len(bolt_final_1)):
            dises=np.linalg.norm(bolt_final_1[i] - bolt_final_2[i],axis=1)
            # print(dises)
            compare=dises<threshed2
            result.append(compare.tolist())
        # print(result)

    else:
        print('两张图角点数都不同')

    return result



if __name__ == '__main__':
    #
    # img1=cv2.imread(r"Datasets\result\target_img\1.png", cv2.IMREAD_GRAYSCALE)
    # # print(img1.shape)
    # img2=cv2.imread(r"Datasets\result\target_img\1.png", cv2.IMREAD_GRAYSCALE)

    img1=cv2.imread(r'E:\train_data2161\002.png', cv2.IMREAD_GRAYSCALE)
    img2=cv2.imread(r'E:\train_data2161\002.png', cv2.IMREAD_GRAYSCALE)

    result=compared(img1,img2)
    print(result)



