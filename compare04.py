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

对compare03做了一点修改
'''

def compared(img1,img2,threshed1=200,threshed2=5):
    index1 = nms(img1)
    # print('index1.shape:',index1.shape)
    index2 = nms(img2)
    # print('index2.shape:',index2.shape)

    #保存结果的列表
    result=[]

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
    #生成len(bolt_index_1)长度的列表
    bolt_new_1=[np.array([[0]])]*len(bolt_index_1)
    # print('bolt_new_1:',bolt_new_1)
    # print('len(bolt_new_1):',len(bolt_new_1))
    bolt_new_2=[np.array([[1000]])]*len(bolt_index_2)
    for i in range(len(bolt_index_1)):
        # print('bolt1:',bolt1)
        # np.linalg.norm(bolt- bolt2,axis=1)求螺钉所有角点各自的欧式距离
        # dis=[np.linalg.norm(bolt- bolt2) if bolt.shape==bolt2.shape else print('维度不同') for bolt2 in bolt_index_2]
        for j in range(len(bolt_index_2)):
            if bolt_index_1[i].shape==bolt_index_2[j].shape:
                # print('np.sum(np.linalg.norm(bolt1- bolt2,axis=1)):',np.sum(np.linalg.norm(bolt1- bolt2,axis=1)))
                if np.sum(np.linalg.norm(bolt_index_1[i]- bolt_index_2[j],axis=1))<threshed1:
                    bolt_new_1[i]=bolt_index_1[i]

                    #有些图半个螺丝没有标注，却被检测出来了，所以测试会比标签图多
                    #严格来说这都是没有错的
                    try:
                        bolt_new_2[i]=bolt_index_2[j]
                    except:
                        print('这图多预测了')

    # print('bolt_new_1:',bolt_new_1)
    # print('bolt_new_2:',bolt_new_2)

    # print('len(bolt_new_1):',len(bolt_new_1))

    '''
    确定对应的角点：也是用欧式距离
    '''
    bolt_final_1 = [np.array([[0]])]*len(bolt_index_1)
    bolt_final_2 = [np.array([[1000]])]*len(bolt_index_2)
    for i in range(len(bolt_new_1)):
        # print('i:',i)
        bolt1=[]
        bolt2=[]
        # print('bolt_new_1[i]:',bolt_new_1[i])
        # print('bolt_new_2[i]:',bolt_new_2[i])
        try:
            for corner1 in bolt_new_1[i]:
                # print('corner1:',corner1)
                    for corner2 in bolt_new_2[i]:
                        # print('corner2:',corner2)
                        dis=np.linalg.norm(corner1 - corner2)
                        # print('dis:',dis)
                        if dis < 5:
                            bolt1.append(corner1.tolist())
                            bolt2.append(corner2.tolist())
            bolt_final_1[i]=np.array(bolt1)
            bolt_final_2[i]=np.array(bolt2)
        except:
            print('这图多预测了')


    # print('bolt_final_1:',bolt_final_1)
    # print('bolt_final_2:',bolt_final_2)


    # print('len(bolt_final_1):',len(bolt_final_1))

    for i in range(len(bolt_final_1)):
        # print('bolt_final_1[i]:',bolt_final_1[i])
        try:
            dises=np.linalg.norm(bolt_final_1[i] - bolt_final_2[i],axis=1)
        # print(dises)
            compare=dises<threshed2
            result.append(compare.tolist())
        except:
            result.append([False])
    # print(result)

    if len(bolt_index_1)<len(bolt_index_2):
        print('此图少检测了螺丝')
        result.append([False])

    return result



if __name__ == '__main__':
    #
    # img1=cv2.imread(r"Datasets\result\target_img\1.png", cv2.IMREAD_GRAYSCALE)
    # # print(img1.shape)
    # img2=cv2.imread(r"Datasets\result\out_img\1.png", cv2.IMREAD_GRAYSCALE)

    img1=cv2.imread(r'Datasets\result\target_img\1.png', cv2.IMREAD_GRAYSCALE)
    img2=cv2.imread(r'Datasets\result\out_img\1.png', cv2.IMREAD_GRAYSCALE)

    result=compared(img1,img2)
    print(result)



