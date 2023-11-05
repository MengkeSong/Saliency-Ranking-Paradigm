import cv2
import numpy as np
from PIL import Image
import os
import h5py

import torch
# print(torch.__version__)
area = 0

path = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/box/salicon_rand/train_box_gt/'  # 输入文件夹地址

gt_name = os.listdir(path)
# gt_name.sort(key=lambda x:int(x[11:-4]))
gt_name.sort(key=lambda x:int(x[:-4]))

count = 0
print(gt_name)
for fn in os.listdir(path):  # fn 表示的是文件名

    count = count + 1

pros = []
index = []
for i in range(0,len(gt_name)):
    # 读取图像

    img1 = cv2.imread(path + gt_name[i] )



    # 变微灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


    gray1_nums = gray1.sum()


    height1 ,width1 = gray1.shape


    gray1_nums_avg = gray1_nums / (height1*width1)


    # gray1_nums_avg = gray1_nums


    # gray1_nums_avg*=1.5
    # gray2_nums_avg*=1.5
    # gray3_nums_avg*=1.5
    # gray4_nums_avg*=1.5
    # gray5_nums_avg*=1.5

    gray1_nums_avg = int(gray1_nums_avg)



    # # gray_avg = (gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg) / 5
    # # gray_avg*=1.5
    # # gray_avg = int(gray_avg)
    #
    # # 大津法二值化
    # # retval, dst = cv2.threshold(gray, 39, 255, cv2.THRESH_BINARY)
    # _, dst1 = cv2.threshold(gray1, gray1_nums_avg, 255, cv2.THRESH_BINARY)
    # _, dst2 = cv2.threshold(gray2, gray2_nums_avg, 255, cv2.THRESH_BINARY)
    # _, dst3 = cv2.threshold(gray3, gray3_nums_avg, 255, cv2.THRESH_BINARY)
    # _, dst4 = cv2.threshold(gray4, gray4_nums_avg, 255, cv2.THRESH_BINARY)
    # _, dst5 = cv2.threshold(gray5, gray5_nums_avg, 255, cv2.THRESH_BINARY)
    #
    # # cv2.imwrite('erzhipic_solo_1_5/'+gt_name[i], dst1)
    # # cv2.imwrite('erzhipic_solo_1_5/'+gt_name[i+1], dst2)
    # # cv2.imwrite('erzhipic_solo_1_5/'+gt_name[i+2], dst3)
    # # cv2.imwrite('erzhipic_solo_1_5/'+gt_name[i+3], dst4)
    # # cv2.imwrite('erzhipic_solo_1_5/'+gt_name[i+4], dst5)
    #
    #
    #
    # # 腐蚀和膨胀是对白色部分而言的，膨胀，白区域变大，最后的参数为迭代次数
    # # dst = cv2.dilate(dst, None, iterations=1)
    # # # 腐蚀，白区域变小
    # # dst = cv2.erode(dst, None, iterations=4)
    # # height, width = dst.shape
    # # nums = height * width
    # area1 = 0
    # for t in range(height1):
    #     for f in range(width1):
    #         if dst1[t, f] == 255:
    #             area1 += 1
    # # if area==height*width :
    # #     area=0
    # # print(pro)
    # print(area1)
    # if width1==1:
    #     area1=0
    # if height1==1:
    #     area1=0
    #
    # area2 = 0
    # for t in range(height2):
    #     for f in range(width2):
    #         if dst2[t, f] == 255:
    #             area2 += 1
    # # if area==height*width :
    # #     area=0
    # # print(pro)
    # print(area2)
    # if width2==1:
    #     area2=0
    # if height2==1:
    #     area2=0
    #
    # area3 = 0
    # for t in range(height3):
    #     for f in range(width3):
    #         if dst3[t, f] == 255:
    #             area3 += 1
    # # if area==height*width :
    # #     area=0
    # # print(pro)
    # print(area3)
    # if width3==1:
    #     area3=0
    # if height3==1:
    #     area3=0
    #
    # area4 = 0
    # for t in range(height4):
    #     for f in range(width4):
    #         if dst4[t, f] == 255:
    #             area4 += 1
    # # if area==height*width :
    # #     area=0
    # # print(pro)
    # print(area4)
    # if width4==1:
    #     area4=0
    # if height4==1:
    #     area4=0
    # area5 = 0
    # for t in range(height5):
    #     for f in range(width5):
    #         if dst5[t, f] == 255:
    #             area5 += 1
    # # if area==height*width :
    # #     area=0
    # # print(pro)
    # print(area5)
    # if width5==1:
    #     area5=0
    # if height5==1:
    #     area5=0
    # # print(area)
    # # if area>=50:
    # #     pro = 1
    # # else:
    # #     pro= 0
    pros.append(gray1_nums_avg)


    # print(pro)
print(len(pros))
# print(pros)
pros = np.array(pros)
# ids = np.argsort(pros)
# print(ids)
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_rand/h5_blcakdot_nums/train_nums.h5', 'w')
f.create_dataset('labels', data=pros)
f.close()



