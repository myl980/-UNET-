# -*- coding: utf-8 -*-

from Graduation_Project.Screws.U_Net.compare04 import compared
import os
import cv2

'''计算测试最终效果'''

out_img_path=r'Datasets\result\out_img'
target_img_path=r'Datasets\result\target_img'


names=os.listdir(out_img_path)
#螺钉预测正确数
correct_num=0.
#螺钉总数
total_num=0.
#螺钉预测错误数
false_num=0.
# print(names)
for name in names:
    print(name)
    out_img = cv2.imread(os.path.join(out_img_path,name), cv2.IMREAD_GRAYSCALE)
    target_img = cv2.imread(os.path.join(target_img_path,name), cv2.IMREAD_GRAYSCALE)

    result=compared(out_img,target_img)
    # print(result)
    for i in result:
        total_num+=1
        num=0
        for k in i:
            if k == True:
                num+=1
                
        #num>4才判定螺钉判定正确
        if num >4:
            correct_num+=1
        else:
            false_num+=1

acc=correct_num/total_num
print('correct_num:',correct_num)
print('total_num:',total_num)
print('false_num:',false_num)
print('acc:',acc)


