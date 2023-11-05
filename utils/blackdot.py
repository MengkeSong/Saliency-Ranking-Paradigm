import cv2
import numpy as np
from PIL import Image
import os
import h5py
# from zhushi import goutou
# goutou()
import torch
# print(torch.__version__)
area = 0

path = '/home/w509/1workspace/lee/360_fix_sort/box/360_new_dataset_ceshi/test_salgan/'  # 输入文件夹地址

gt_name = os.listdir(path)
# gt_name.sort(key=lambda x:int(x[11:-4]))
gt_name.sort(key=lambda x:int(x[:-4]))

count = 0
print(gt_name)
for fn in os.listdir(path):  # fn 表示的是文件名

    count = count + 1

pros = []
index = []
for i in range(len(gt_name)):
    # 读取图像

    img = cv2.imread(path + gt_name[i] )

    # 变微灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_nums = gray.sum()
    height,width = gray.shape
    total = height * width
    gray_avg = gray_nums / total
    # 大津法二值化
    # retval, dst = cv2.threshold(gray, 39, 255, cv2.THRESH_BINARY)
    # retval, dst = cv2.threshold(gray, 23, 255, cv2.THRESH_BINARY)
    #
    #
    # # 腐蚀和膨胀是对白色部分而言的，膨胀，白区域变大，最后的参数为迭代次数
    # # dst = cv2.dilate(dst, None, iterations=1)
    # # # 腐蚀，白区域变小
    # # dst = cv2.erode(dst, None, iterations=4)
    # height, width = dst.shape
    # nums = height * width
    # area = 0
    # for t in range(height):
    #     for f in range(width):
    #         if dst[t, f] == 255:
    #             area += 1
    # # if area==height*width :
    # #     area=0
    # # print(pro)
    # print(area)
    # if width==1:
    #     area=0
    # if height==1:
    #     area=0
    # print(area)
    # if area>=50:
    #     pro = 1
    # else:
    #     pro= 0
    pros.append(gray_avg)
    # print(pro)
print(len(pros))
# print(pros)
pros = np.array(pros)
# ids = np.argsort(pros)
# print(ids)
f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/feature/360_new_dataset/h5_blackdot_nums/test_nums_sal_salgan_gray_avg.h5', 'w')
f.create_dataset('labels', data=pros)
f.close()



