import cv2
import numpy as np
from PIL import Image
import os
import h5py

# from zhushi import goutou
# goutou()

area = 0
global ids
ids = 0
pros = []
path1 = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/box/salicon/salgan/val/'  # 输入文件夹地址

gt_name = os.listdir(path1)
# gt_name.sort(key=lambda x:int(x[11:-4]))
gt_name.sort(key=lambda x: int(x[-10:-4]))

count = 0
print(gt_name)

path = r'/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/boxtxt/vallocal/'
imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x: int(x[:-8]))
print(imgs_path)

for name in imgs_path:
    txt_path = path + name

    f = open(txt_path, "r")
    length = 0
    while True:
        line = f.readline()
        if line:
            length += 1
        else:
            break
    if length == 3:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])

        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()

        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape

        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)

        gray_avg = (gray1_nums_avg + gray2_nums_avg + gray3_nums_avg) / 3
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)

        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)

        ids += 3
    if length == 4:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])

        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()

        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape

        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)

        gray_avg = (gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg) / 4
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)

        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)

        ids += 4
    if length == 5:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])

        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()

        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape

        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)

        gray_avg = (gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg) / 5
        gray_avg *= 0.7
        gray_avg = int(gray_avg)

        # 大津法二值化
        # retval, dst = cv2.threshold(gray, 39, 255, cv2.THRESH_BINARY)
        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)

        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1
        # if area==height*width :
        #     area=0
        # print(pro)
        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        # if area==height*width :
        #     area=0
        # print(pro)
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        # if area==height*width :
        #     area=0
        # print(pro)
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        # if area==height*width :
        #     area=0
        # print(pro)
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0
        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        # if area==height*width :
        #     area=0
        # print(pro)
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0
        # print(area)
        # if area>=50:
        #     pro = 1
        # else:
        #     pro= 0
        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)

        ids += 5
    if length == 6:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])

        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)

        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()

        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape

        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)

        gray_avg = (gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg+ gray5_nums_avg + gray6_nums_avg) / 6
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)


        ids += 6
    if length == 7:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape

        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)

        gray_avg = (
                               gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg+ gray7_nums_avg) / 7
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)

        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)

        ids += 7
    if length == 8:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])

        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()

        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape

        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)

        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg+ gray8_nums_avg) / 8
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)

        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)

        ids += 8
    if length == 9:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)

        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg+ gray9_nums_avg) / 9
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)

        ids += 9
    if length == 10:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)

        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape

        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)

        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                   + gray10_nums_avg) / 10
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)

        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)


        ids += 10
    if length == 11:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg+ gray11_nums_avg) / 11
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)

        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0
        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)


        ids += 11
    if length == 12:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])
        img12 = cv2.imread(path1 + gt_name[ids + 11])



        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        gray12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()
        gray12_nums = gray12.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape
        height12, width12 = gray12.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)
        gray12_nums_avg = gray12_nums / (height12 * width12)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg+ gray11_nums_avg+ gray12_nums_avg) / 12
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst12 = cv2.threshold(gray12, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0

        area12 = 0
        for t in range(height12):
            for f in range(width12):
                if dst12[t, f] == 255:
                    area12 += 1
        print(area12)
        if width12 == 1:
            area12 = 0
        if height12 == 1:
            area12 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)
        pros.append(area12)


        ids += 12
    if length == 13:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])
        img12 = cv2.imread(path1 + gt_name[ids + 11])
        img13 = cv2.imread(path1 + gt_name[ids + 12])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        gray12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
        gray13 = cv2.cvtColor(img13, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()
        gray12_nums = gray12.sum()
        gray13_nums = gray13.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape
        height12, width12 = gray12.shape
        height13, width13 = gray13.shape

        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)
        gray12_nums_avg = gray12_nums / (height12 * width12)
        gray13_nums_avg = gray13_nums / (height13 * width13)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg + gray11_nums_avg + gray12_nums_avg+ gray13_nums_avg) / 13
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst12 = cv2.threshold(gray12, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst13 = cv2.threshold(gray13, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0

        area12 = 0
        for t in range(height12):
            for f in range(width12):
                if dst12[t, f] == 255:
                    area12 += 1
        print(area12)
        if width12 == 1:
            area12 = 0
        if height12 == 1:
            area12 = 0

        area13 = 0
        for t in range(height13):
            for f in range(width13):
                if dst13[t, f] == 255:
                    area13 += 1
        print(area13)
        if width13 == 1:
            area13 = 0
        if height13 == 1:
            area13 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)
        pros.append(area12)
        pros.append(area13)

        ids += 13
    if length == 14:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])
        img12 = cv2.imread(path1 + gt_name[ids + 11])
        img13 = cv2.imread(path1 + gt_name[ids + 12])
        img14 = cv2.imread(path1 + gt_name[ids + 13])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        gray12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
        gray13 = cv2.cvtColor(img13, cv2.COLOR_BGR2GRAY)
        gray14 = cv2.cvtColor(img14, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()
        gray12_nums = gray12.sum()
        gray13_nums = gray13.sum()
        gray14_nums = gray14.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape
        height12, width12 = gray12.shape
        height13, width13 = gray13.shape
        height14, width14 = gray14.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)
        gray12_nums_avg = gray12_nums / (height12 * width12)
        gray13_nums_avg = gray13_nums / (height13 * width13)
        gray14_nums_avg = gray14_nums / (height14 * width14)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg + gray11_nums_avg + gray12_nums_avg + gray13_nums_avg + gray14_nums_avg) / 14
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst12 = cv2.threshold(gray12, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst13 = cv2.threshold(gray13, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst14 = cv2.threshold(gray14, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0

        area12 = 0
        for t in range(height12):
            for f in range(width12):
                if dst12[t, f] == 255:
                    area12 += 1
        print(area12)
        if width12 == 1:
            area12 = 0
        if height12 == 1:
            area12 = 0

        area13 = 0
        for t in range(height13):
            for f in range(width13):
                if dst13[t, f] == 255:
                    area13 += 1
        print(area13)
        if width13 == 1:
            area13 = 0
        if height13 == 1:
            area13 = 0

        area14 = 0
        for t in range(height14):
            for f in range(width14):
                if dst14[t, f] == 255:
                    area14 += 1
        print(area14)
        if width14 == 1:
            area14 = 0
        if height14 == 1:
            area14 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)
        pros.append(area12)
        pros.append(area13)
        pros.append(area14)


        ids += 14
    if length == 15:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])
        img12 = cv2.imread(path1 + gt_name[ids + 11])
        img13 = cv2.imread(path1 + gt_name[ids + 12])
        img14 = cv2.imread(path1 + gt_name[ids + 13])
        img15 = cv2.imread(path1 + gt_name[ids + 14])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        gray12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
        gray13 = cv2.cvtColor(img13, cv2.COLOR_BGR2GRAY)
        gray14 = cv2.cvtColor(img14, cv2.COLOR_BGR2GRAY)
        gray15 = cv2.cvtColor(img15, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()
        gray12_nums = gray12.sum()
        gray13_nums = gray13.sum()
        gray14_nums = gray14.sum()
        gray15_nums = gray15.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape
        height12, width12 = gray12.shape
        height13, width13 = gray13.shape
        height14, width14 = gray14.shape
        height15, width15 = gray15.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)
        gray12_nums_avg = gray12_nums / (height12 * width12)
        gray13_nums_avg = gray13_nums / (height13 * width13)
        gray14_nums_avg = gray14_nums / (height14 * width14)
        gray15_nums_avg = gray15_nums / (height15 * width15)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg + gray11_nums_avg + gray12_nums_avg + gray13_nums_avg + gray14_nums_avg+ gray15_nums_avg) / 15
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst12 = cv2.threshold(gray12, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst13 = cv2.threshold(gray13, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst14 = cv2.threshold(gray14, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst15 = cv2.threshold(gray15, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0

        area12 = 0
        for t in range(height12):
            for f in range(width12):
                if dst12[t, f] == 255:
                    area12 += 1
        print(area12)
        if width12 == 1:
            area12 = 0
        if height12 == 1:
            area12 = 0

        area13 = 0
        for t in range(height13):
            for f in range(width13):
                if dst13[t, f] == 255:
                    area13 += 1
        print(area13)
        if width13 == 1:
            area13 = 0
        if height13 == 1:
            area13 = 0

        area14 = 0
        for t in range(height14):
            for f in range(width14):
                if dst14[t, f] == 255:
                    area14 += 1
        print(area14)
        if width14 == 1:
            area14 = 0
        if height14 == 1:
            area14 = 0

        area15 = 0
        for t in range(height15):
            for f in range(width15):
                if dst15[t, f] == 255:
                    area15 += 1
        print(area15)
        if width15 == 1:
            area15 = 0
        if height15 == 1:
            area15 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)
        pros.append(area12)
        pros.append(area13)
        pros.append(area14)
        pros.append(area15)

        ids += 15
    if length == 16:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])
        img12 = cv2.imread(path1 + gt_name[ids + 11])
        img13 = cv2.imread(path1 + gt_name[ids + 12])
        img14 = cv2.imread(path1 + gt_name[ids + 13])
        img15 = cv2.imread(path1 + gt_name[ids + 14])
        img16 = cv2.imread(path1 + gt_name[ids + 15])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        gray12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
        gray13 = cv2.cvtColor(img13, cv2.COLOR_BGR2GRAY)
        gray14 = cv2.cvtColor(img14, cv2.COLOR_BGR2GRAY)
        gray15 = cv2.cvtColor(img15, cv2.COLOR_BGR2GRAY)
        gray16 = cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()
        gray12_nums = gray12.sum()
        gray13_nums = gray13.sum()
        gray14_nums = gray14.sum()
        gray15_nums = gray15.sum()
        gray16_nums = gray16.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape
        height12, width12 = gray12.shape
        height13, width13 = gray13.shape
        height14, width14 = gray14.shape
        height15, width15 = gray15.shape
        height16, width16 = gray16.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)
        gray12_nums_avg = gray12_nums / (height12 * width12)
        gray13_nums_avg = gray13_nums / (height13 * width13)
        gray14_nums_avg = gray14_nums / (height14 * width14)
        gray15_nums_avg = gray15_nums / (height15 * width15)
        gray16_nums_avg = gray16_nums / (height16 * width16)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg + gray11_nums_avg + gray12_nums_avg + gray13_nums_avg + gray14_nums_avg + gray15_nums_avg+ gray16_nums_avg) / 16
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst12 = cv2.threshold(gray12, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst13 = cv2.threshold(gray13, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst14 = cv2.threshold(gray14, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst15 = cv2.threshold(gray15, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst16 = cv2.threshold(gray16, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0

        area12 = 0
        for t in range(height12):
            for f in range(width12):
                if dst12[t, f] == 255:
                    area12 += 1
        print(area12)
        if width12 == 1:
            area12 = 0
        if height12 == 1:
            area12 = 0

        area13 = 0
        for t in range(height13):
            for f in range(width13):
                if dst13[t, f] == 255:
                    area13 += 1
        print(area13)
        if width13 == 1:
            area13 = 0
        if height13 == 1:
            area13 = 0

        area14 = 0
        for t in range(height14):
            for f in range(width14):
                if dst14[t, f] == 255:
                    area14 += 1
        print(area14)
        if width14 == 1:
            area14 = 0
        if height14 == 1:
            area14 = 0

        area15 = 0
        for t in range(height15):
            for f in range(width15):
                if dst15[t, f] == 255:
                    area15 += 1
        print(area15)
        if width15 == 1:
            area15 = 0
        if height15 == 1:
            area15 = 0

        area16 = 0
        for t in range(height16):
            for f in range(width16):
                if dst16[t, f] == 255:
                    area16 += 1
        print(area16)
        if width16 == 1:
            area16 = 0
        if height16 == 1:
            area16 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)
        pros.append(area12)
        pros.append(area13)
        pros.append(area14)
        pros.append(area15)
        pros.append(area16)


        ids += 16
    if length == 17:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])
        img12 = cv2.imread(path1 + gt_name[ids + 11])
        img13 = cv2.imread(path1 + gt_name[ids + 12])
        img14 = cv2.imread(path1 + gt_name[ids + 13])
        img15 = cv2.imread(path1 + gt_name[ids + 14])
        img16 = cv2.imread(path1 + gt_name[ids + 15])
        img17 = cv2.imread(path1 + gt_name[ids + 16])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        gray12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
        gray13 = cv2.cvtColor(img13, cv2.COLOR_BGR2GRAY)
        gray14 = cv2.cvtColor(img14, cv2.COLOR_BGR2GRAY)
        gray15 = cv2.cvtColor(img15, cv2.COLOR_BGR2GRAY)
        gray16 = cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)
        gray17 = cv2.cvtColor(img17, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()
        gray12_nums = gray12.sum()
        gray13_nums = gray13.sum()
        gray14_nums = gray14.sum()
        gray15_nums = gray15.sum()
        gray16_nums = gray16.sum()
        gray17_nums = gray17.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape
        height12, width12 = gray12.shape
        height13, width13 = gray13.shape
        height14, width14 = gray14.shape
        height15, width15 = gray15.shape
        height16, width16 = gray16.shape
        height17, width17 = gray17.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)
        gray12_nums_avg = gray12_nums / (height12 * width12)
        gray13_nums_avg = gray13_nums / (height13 * width13)
        gray14_nums_avg = gray14_nums / (height14 * width14)
        gray15_nums_avg = gray15_nums / (height15 * width15)
        gray16_nums_avg = gray16_nums / (height16 * width16)
        gray17_nums_avg = gray17_nums / (height17 * width17)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg + gray11_nums_avg + gray12_nums_avg + gray13_nums_avg + gray14_nums_avg + gray15_nums_avg + gray16_nums_avg +gray17_nums_avg) / 17
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst12 = cv2.threshold(gray12, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst13 = cv2.threshold(gray13, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst14 = cv2.threshold(gray14, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst15 = cv2.threshold(gray15, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst16 = cv2.threshold(gray16, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst17 = cv2.threshold(gray17, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0

        area12 = 0
        for t in range(height12):
            for f in range(width12):
                if dst12[t, f] == 255:
                    area12 += 1
        print(area12)
        if width12 == 1:
            area12 = 0
        if height12 == 1:
            area12 = 0

        area13 = 0
        for t in range(height13):
            for f in range(width13):
                if dst13[t, f] == 255:
                    area13 += 1
        print(area13)
        if width13 == 1:
            area13 = 0
        if height13 == 1:
            area13 = 0

        area14 = 0
        for t in range(height14):
            for f in range(width14):
                if dst14[t, f] == 255:
                    area14 += 1
        print(area14)
        if width14 == 1:
            area14 = 0
        if height14 == 1:
            area14 = 0

        area15 = 0
        for t in range(height15):
            for f in range(width15):
                if dst15[t, f] == 255:
                    area15 += 1
        print(area15)
        if width15 == 1:
            area15 = 0
        if height15 == 1:
            area15 = 0

        area16 = 0
        for t in range(height16):
            for f in range(width16):
                if dst16[t, f] == 255:
                    area16 += 1
        print(area16)
        if width16 == 1:
            area16 = 0
        if height16 == 1:
            area16 = 0

        area17 = 0
        for t in range(height17):
            for f in range(width17):
                if dst17[t, f] == 255:
                    area17 += 1
        print(area17)
        if width17 == 1:
            area17 = 0
        if height17 == 1:
            area17 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)
        pros.append(area12)
        pros.append(area13)
        pros.append(area14)
        pros.append(area15)
        pros.append(area16)
        pros.append(area17)


        ids += 17
    if length == 18:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])
        img12 = cv2.imread(path1 + gt_name[ids + 11])
        img13 = cv2.imread(path1 + gt_name[ids + 12])
        img14 = cv2.imread(path1 + gt_name[ids + 13])
        img15 = cv2.imread(path1 + gt_name[ids + 14])
        img16 = cv2.imread(path1 + gt_name[ids + 15])
        img17 = cv2.imread(path1 + gt_name[ids + 16])
        img18 = cv2.imread(path1 + gt_name[ids + 17])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        gray12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
        gray13 = cv2.cvtColor(img13, cv2.COLOR_BGR2GRAY)
        gray14 = cv2.cvtColor(img14, cv2.COLOR_BGR2GRAY)
        gray15 = cv2.cvtColor(img15, cv2.COLOR_BGR2GRAY)
        gray16 = cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)
        gray17 = cv2.cvtColor(img17, cv2.COLOR_BGR2GRAY)
        gray18 = cv2.cvtColor(img18, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()
        gray12_nums = gray12.sum()
        gray13_nums = gray13.sum()
        gray14_nums = gray14.sum()
        gray15_nums = gray15.sum()
        gray16_nums = gray16.sum()
        gray17_nums = gray17.sum()
        gray18_nums = gray18.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape
        height12, width12 = gray12.shape
        height13, width13 = gray13.shape
        height14, width14 = gray14.shape
        height15, width15 = gray15.shape
        height16, width16 = gray16.shape
        height17, width17 = gray17.shape
        height18, width18 = gray18.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)
        gray12_nums_avg = gray12_nums / (height12 * width12)
        gray13_nums_avg = gray13_nums / (height13 * width13)
        gray14_nums_avg = gray14_nums / (height14 * width14)
        gray15_nums_avg = gray15_nums / (height15 * width15)
        gray16_nums_avg = gray16_nums / (height16 * width16)
        gray17_nums_avg = gray17_nums / (height17 * width17)
        gray18_nums_avg = gray18_nums / (height18 * width18)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg + gray11_nums_avg + gray12_nums_avg + gray13_nums_avg + gray14_nums_avg + gray15_nums_avg + gray16_nums_avg + gray17_nums_avg
                   + gray18_nums_avg) / 18
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst12 = cv2.threshold(gray12, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst13 = cv2.threshold(gray13, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst14 = cv2.threshold(gray14, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst15 = cv2.threshold(gray15, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst16 = cv2.threshold(gray16, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst17 = cv2.threshold(gray17, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst18 = cv2.threshold(gray18, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0

        area12 = 0
        for t in range(height12):
            for f in range(width12):
                if dst12[t, f] == 255:
                    area12 += 1
        print(area12)
        if width12 == 1:
            area12 = 0
        if height12 == 1:
            area12 = 0

        area13 = 0
        for t in range(height13):
            for f in range(width13):
                if dst13[t, f] == 255:
                    area13 += 1
        print(area13)
        if width13 == 1:
            area13 = 0
        if height13 == 1:
            area13 = 0

        area14 = 0
        for t in range(height14):
            for f in range(width14):
                if dst14[t, f] == 255:
                    area14 += 1
        print(area14)
        if width14 == 1:
            area14 = 0
        if height14 == 1:
            area14 = 0

        area15 = 0
        for t in range(height15):
            for f in range(width15):
                if dst15[t, f] == 255:
                    area15 += 1
        print(area15)
        if width15 == 1:
            area15 = 0
        if height15 == 1:
            area15 = 0

        area16 = 0
        for t in range(height16):
            for f in range(width16):
                if dst16[t, f] == 255:
                    area16 += 1
        print(area16)
        if width16 == 1:
            area16 = 0
        if height16 == 1:
            area16 = 0

        area17 = 0
        for t in range(height17):
            for f in range(width17):
                if dst17[t, f] == 255:
                    area17 += 1
        print(area17)
        if width17 == 1:
            area17 = 0
        if height17 == 1:
            area17 = 0

        area18 = 0
        for t in range(height18):
            for f in range(width18):
                if dst18[t, f] == 255:
                    area18 += 1
        print(area18)
        if width18 == 1:
            area18 = 0
        if height18 == 1:
            area18 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)
        pros.append(area12)
        pros.append(area13)
        pros.append(area14)
        pros.append(area15)
        pros.append(area16)
        pros.append(area17)
        pros.append(area18)


        ids += 18
    if length == 19:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])
        img12 = cv2.imread(path1 + gt_name[ids + 11])
        img13 = cv2.imread(path1 + gt_name[ids + 12])
        img14 = cv2.imread(path1 + gt_name[ids + 13])
        img15 = cv2.imread(path1 + gt_name[ids + 14])
        img16 = cv2.imread(path1 + gt_name[ids + 15])
        img17 = cv2.imread(path1 + gt_name[ids + 16])
        img18 = cv2.imread(path1 + gt_name[ids + 17])
        img19 = cv2.imread(path1 + gt_name[ids + 18])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        gray12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
        gray13 = cv2.cvtColor(img13, cv2.COLOR_BGR2GRAY)
        gray14 = cv2.cvtColor(img14, cv2.COLOR_BGR2GRAY)
        gray15 = cv2.cvtColor(img15, cv2.COLOR_BGR2GRAY)
        gray16 = cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)
        gray17 = cv2.cvtColor(img17, cv2.COLOR_BGR2GRAY)
        gray18 = cv2.cvtColor(img18, cv2.COLOR_BGR2GRAY)
        gray19 = cv2.cvtColor(img19, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()
        gray12_nums = gray12.sum()
        gray13_nums = gray13.sum()
        gray14_nums = gray14.sum()
        gray15_nums = gray15.sum()
        gray16_nums = gray16.sum()
        gray17_nums = gray17.sum()
        gray18_nums = gray18.sum()
        gray19_nums = gray19.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape
        height12, width12 = gray12.shape
        height13, width13 = gray13.shape
        height14, width14 = gray14.shape
        height15, width15 = gray15.shape
        height16, width16 = gray16.shape
        height17, width17 = gray17.shape
        height18, width18 = gray18.shape
        height19, width19 = gray19.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)
        gray12_nums_avg = gray12_nums / (height12 * width12)
        gray13_nums_avg = gray13_nums / (height13 * width13)
        gray14_nums_avg = gray14_nums / (height14 * width14)
        gray15_nums_avg = gray15_nums / (height15 * width15)
        gray16_nums_avg = gray16_nums / (height16 * width16)
        gray17_nums_avg = gray17_nums / (height17 * width17)
        gray18_nums_avg = gray18_nums / (height18 * width18)
        gray19_nums_avg = gray19_nums / (height19 * width19)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg + gray11_nums_avg + gray12_nums_avg + gray13_nums_avg + gray14_nums_avg + gray15_nums_avg + gray16_nums_avg + gray17_nums_avg
                           + gray18_nums_avg + gray19_nums_avg) / 19
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst12 = cv2.threshold(gray12, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst13 = cv2.threshold(gray13, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst14 = cv2.threshold(gray14, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst15 = cv2.threshold(gray15, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst16 = cv2.threshold(gray16, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst17 = cv2.threshold(gray17, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst18 = cv2.threshold(gray18, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst19 = cv2.threshold(gray19, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0

        area12 = 0
        for t in range(height12):
            for f in range(width12):
                if dst12[t, f] == 255:
                    area12 += 1
        print(area12)
        if width12 == 1:
            area12 = 0
        if height12 == 1:
            area12 = 0

        area13 = 0
        for t in range(height13):
            for f in range(width13):
                if dst13[t, f] == 255:
                    area13 += 1
        print(area13)
        if width13 == 1:
            area13 = 0
        if height13 == 1:
            area13 = 0

        area14 = 0
        for t in range(height14):
            for f in range(width14):
                if dst14[t, f] == 255:
                    area14 += 1
        print(area14)
        if width14 == 1:
            area14 = 0
        if height14 == 1:
            area14 = 0

        area15 = 0
        for t in range(height15):
            for f in range(width15):
                if dst15[t, f] == 255:
                    area15 += 1
        print(area15)
        if width15 == 1:
            area15 = 0
        if height15 == 1:
            area15 = 0

        area16 = 0
        for t in range(height16):
            for f in range(width16):
                if dst16[t, f] == 255:
                    area16 += 1
        print(area16)
        if width16 == 1:
            area16 = 0
        if height16 == 1:
            area16 = 0

        area17 = 0
        for t in range(height17):
            for f in range(width17):
                if dst17[t, f] == 255:
                    area17 += 1
        print(area17)
        if width17 == 1:
            area17 = 0
        if height17 == 1:
            area17 = 0

        area18 = 0
        for t in range(height18):
            for f in range(width18):
                if dst18[t, f] == 255:
                    area18 += 1
        print(area18)
        if width18 == 1:
            area18 = 0
        if height18 == 1:
            area18 = 0

        area19 = 0
        for t in range(height19):
            for f in range(width19):
                if dst19[t, f] == 255:
                    area19 += 1
        print(area19)
        if width19 == 1:
            area19 = 0
        if height19 == 1:
            area19 = 0



        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)
        pros.append(area12)
        pros.append(area13)
        pros.append(area14)
        pros.append(area15)
        pros.append(area16)
        pros.append(area17)
        pros.append(area18)
        pros.append(area19)


        ids += 19
    if length == 20:
        # 读取图像

        img1 = cv2.imread(path1 + gt_name[ids])
        img2 = cv2.imread(path1 + gt_name[ids + 1])
        img3 = cv2.imread(path1 + gt_name[ids + 2])
        img4 = cv2.imread(path1 + gt_name[ids + 3])
        img5 = cv2.imread(path1 + gt_name[ids + 4])
        img6 = cv2.imread(path1 + gt_name[ids + 5])
        img7 = cv2.imread(path1 + gt_name[ids + 6])
        img8 = cv2.imread(path1 + gt_name[ids + 7])
        img9 = cv2.imread(path1 + gt_name[ids + 8])
        img10 = cv2.imread(path1 + gt_name[ids + 9])
        img11 = cv2.imread(path1 + gt_name[ids + 10])
        img12 = cv2.imread(path1 + gt_name[ids + 11])
        img13 = cv2.imread(path1 + gt_name[ids + 12])
        img14 = cv2.imread(path1 + gt_name[ids + 13])
        img15 = cv2.imread(path1 + gt_name[ids + 14])
        img16 = cv2.imread(path1 + gt_name[ids + 15])
        img17 = cv2.imread(path1 + gt_name[ids + 16])
        img18 = cv2.imread(path1 + gt_name[ids + 17])
        img19 = cv2.imread(path1 + gt_name[ids + 18])
        img20 = cv2.imread(path1 + gt_name[ids + 19])


        # 变微灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
        gray5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        gray6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
        gray7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
        gray8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        gray9 = cv2.cvtColor(img9, cv2.COLOR_BGR2GRAY)
        gray10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
        gray11 = cv2.cvtColor(img11, cv2.COLOR_BGR2GRAY)
        gray12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
        gray13 = cv2.cvtColor(img13, cv2.COLOR_BGR2GRAY)
        gray14 = cv2.cvtColor(img14, cv2.COLOR_BGR2GRAY)
        gray15 = cv2.cvtColor(img15, cv2.COLOR_BGR2GRAY)
        gray16 = cv2.cvtColor(img16, cv2.COLOR_BGR2GRAY)
        gray17 = cv2.cvtColor(img17, cv2.COLOR_BGR2GRAY)
        gray18 = cv2.cvtColor(img18, cv2.COLOR_BGR2GRAY)
        gray19 = cv2.cvtColor(img19, cv2.COLOR_BGR2GRAY)
        gray20 = cv2.cvtColor(img20, cv2.COLOR_BGR2GRAY)


        gray1_nums = gray1.sum()
        gray2_nums = gray2.sum()
        gray3_nums = gray3.sum()
        gray4_nums = gray4.sum()
        gray5_nums = gray5.sum()
        gray6_nums = gray6.sum()
        gray7_nums = gray7.sum()
        gray8_nums = gray8.sum()
        gray9_nums = gray9.sum()
        gray10_nums = gray10.sum()
        gray11_nums = gray11.sum()
        gray12_nums = gray12.sum()
        gray13_nums = gray13.sum()
        gray14_nums = gray14.sum()
        gray15_nums = gray15.sum()
        gray16_nums = gray16.sum()
        gray17_nums = gray17.sum()
        gray18_nums = gray18.sum()
        gray19_nums = gray19.sum()
        gray20_nums = gray20.sum()


        height1, width1 = gray1.shape
        height2, width2 = gray2.shape
        height3, width3 = gray3.shape
        height4, width4 = gray4.shape
        height5, width5 = gray5.shape
        height6, width6 = gray6.shape
        height7, width7 = gray7.shape
        height8, width8 = gray8.shape
        height9, width9 = gray9.shape
        height10, width10 = gray10.shape
        height11, width11 = gray11.shape
        height12, width12 = gray12.shape
        height13, width13 = gray13.shape
        height14, width14 = gray14.shape
        height15, width15 = gray15.shape
        height16, width16 = gray16.shape
        height17, width17 = gray17.shape
        height18, width18 = gray18.shape
        height19, width19 = gray19.shape
        height20, width20 = gray20.shape


        gray1_nums_avg = gray1_nums / (height1 * width1)
        gray2_nums_avg = gray2_nums / (height2 * width2)
        gray3_nums_avg = gray3_nums / (height3 * width3)
        gray4_nums_avg = gray4_nums / (height4 * width4)
        gray5_nums_avg = gray5_nums / (height5 * width5)
        gray6_nums_avg = gray6_nums / (height6 * width6)
        gray7_nums_avg = gray7_nums / (height7 * width7)
        gray8_nums_avg = gray8_nums / (height8 * width8)
        gray9_nums_avg = gray9_nums / (height9 * width9)
        gray10_nums_avg = gray10_nums / (height10 * width10)
        gray11_nums_avg = gray11_nums / (height11 * width11)
        gray12_nums_avg = gray12_nums / (height12 * width12)
        gray13_nums_avg = gray13_nums / (height13 * width13)
        gray14_nums_avg = gray14_nums / (height14 * width14)
        gray15_nums_avg = gray15_nums / (height15 * width15)
        gray16_nums_avg = gray16_nums / (height16 * width16)
        gray17_nums_avg = gray17_nums / (height17 * width17)
        gray18_nums_avg = gray18_nums / (height18 * width18)
        gray19_nums_avg = gray19_nums / (height19 * width19)
        gray20_nums_avg = gray20_nums / (height20 * width20)


        gray_avg = (
                           gray1_nums_avg + gray2_nums_avg + gray3_nums_avg + gray4_nums_avg + gray5_nums_avg + gray6_nums_avg + gray7_nums_avg + gray8_nums_avg + gray9_nums_avg
                           + gray10_nums_avg + gray11_nums_avg + gray12_nums_avg + gray13_nums_avg + gray14_nums_avg + gray15_nums_avg + gray16_nums_avg + gray17_nums_avg
                           + gray18_nums_avg + gray19_nums_avg+ gray20_nums_avg) / 20
        gray_avg *= 1.5
        gray_avg = int(gray_avg)

        # 大津法二值化

        _, dst1 = cv2.threshold(gray1, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst2 = cv2.threshold(gray2, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst3 = cv2.threshold(gray3, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst4 = cv2.threshold(gray4, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst5 = cv2.threshold(gray5, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst6 = cv2.threshold(gray6, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst7 = cv2.threshold(gray7, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst8 = cv2.threshold(gray8, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst9 = cv2.threshold(gray9, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst10 = cv2.threshold(gray10, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst11 = cv2.threshold(gray11, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst12 = cv2.threshold(gray12, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst13 = cv2.threshold(gray13, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst14 = cv2.threshold(gray14, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst15 = cv2.threshold(gray15, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst16 = cv2.threshold(gray16, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst17 = cv2.threshold(gray17, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst18 = cv2.threshold(gray18, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst19 = cv2.threshold(gray19, gray_avg, 255, cv2.THRESH_BINARY)
        _, dst20 = cv2.threshold(gray20, gray_avg, 255, cv2.THRESH_BINARY)


        area1 = 0
        for t in range(height1):
            for f in range(width1):
                if dst1[t, f] == 255:
                    area1 += 1

        print(area1)
        if width1 == 1:
            area1 = 0
        if height1 == 1:
            area1 = 0

        area2 = 0
        for t in range(height2):
            for f in range(width2):
                if dst2[t, f] == 255:
                    area2 += 1
        print(area2)
        if width2 == 1:
            area2 = 0
        if height2 == 1:
            area2 = 0

        area3 = 0
        for t in range(height3):
            for f in range(width3):
                if dst3[t, f] == 255:
                    area3 += 1
        print(area3)
        if width3 == 1:
            area3 = 0
        if height3 == 1:
            area3 = 0

        area4 = 0
        for t in range(height4):
            for f in range(width4):
                if dst4[t, f] == 255:
                    area4 += 1
        print(area4)
        if width4 == 1:
            area4 = 0
        if height4 == 1:
            area4 = 0

        area5 = 0
        for t in range(height5):
            for f in range(width5):
                if dst5[t, f] == 255:
                    area5 += 1
        print(area5)
        if width5 == 1:
            area5 = 0
        if height5 == 1:
            area5 = 0

        area6 = 0
        for t in range(height6):
            for f in range(width6):
                if dst6[t, f] == 255:
                    area6 += 1
        print(area6)
        if width6 == 1:
            area6 = 0
        if height6 == 1:
            area6 = 0

        area7 = 0
        for t in range(height7):
            for f in range(width7):
                if dst7[t, f] == 255:
                    area7 += 1
        print(area7)
        if width7 == 1:
            area7 = 0
        if height7 == 1:
            area7 = 0

        area8 = 0
        for t in range(height8):
            for f in range(width8):
                if dst8[t, f] == 255:
                    area8 += 1
        print(area8)
        if width8 == 1:
            area8 = 0
        if height8 == 1:
            area8 = 0

        area9 = 0
        for t in range(height9):
            for f in range(width9):
                if dst9[t, f] == 255:
                    area9 += 1
        print(area9)
        if width9 == 1:
            area9 = 0
        if height9 == 1:
            area9 = 0

        area10 = 0
        for t in range(height10):
            for f in range(width10):
                if dst10[t, f] == 255:
                    area10 += 1
        print(area10)
        if width10 == 1:
            area10 = 0
        if height10 == 1:
            area10 = 0

        area11 = 0
        for t in range(height11):
            for f in range(width11):
                if dst11[t, f] == 255:
                    area11 += 1
        print(area11)
        if width11 == 1:
            area11 = 0
        if height11 == 1:
            area11 = 0

        area12 = 0
        for t in range(height12):
            for f in range(width12):
                if dst12[t, f] == 255:
                    area12 += 1
        print(area12)
        if width12 == 1:
            area12 = 0
        if height12 == 1:
            area12 = 0

        area13 = 0
        for t in range(height13):
            for f in range(width13):
                if dst13[t, f] == 255:
                    area13 += 1
        print(area13)
        if width13 == 1:
            area13 = 0
        if height13 == 1:
            area13 = 0

        area14 = 0
        for t in range(height14):
            for f in range(width14):
                if dst14[t, f] == 255:
                    area14 += 1
        print(area14)
        if width14 == 1:
            area14 = 0
        if height14 == 1:
            area14 = 0

        area15 = 0
        for t in range(height15):
            for f in range(width15):
                if dst15[t, f] == 255:
                    area15 += 1
        print(area15)
        if width15 == 1:
            area15 = 0
        if height15 == 1:
            area15 = 0

        area16 = 0
        for t in range(height16):
            for f in range(width16):
                if dst16[t, f] == 255:
                    area16 += 1
        print(area16)
        if width16 == 1:
            area16 = 0
        if height16 == 1:
            area16 = 0

        area17 = 0
        for t in range(height17):
            for f in range(width17):
                if dst17[t, f] == 255:
                    area17 += 1
        print(area17)
        if width17 == 1:
            area17 = 0
        if height17 == 1:
            area17 = 0

        area18 = 0
        for t in range(height18):
            for f in range(width18):
                if dst18[t, f] == 255:
                    area18 += 1
        print(area18)
        if width18 == 1:
            area18 = 0
        if height18 == 1:
            area18 = 0

        area19 = 0
        for t in range(height19):
            for f in range(width19):
                if dst19[t, f] == 255:
                    area19 += 1
        print(area19)
        if width19 == 1:
            area19 = 0
        if height19 == 1:
            area19 = 0

        area20 = 0
        for t in range(height20):
            for f in range(width20):
                if dst20[t, f] == 255:
                    area20 += 1
        print(area20)
        if width20 == 1:
            area20 = 0
        if height20 == 1:
            area20 = 0

        pros.append(area1)
        pros.append(area2)
        pros.append(area3)
        pros.append(area4)
        pros.append(area5)
        pros.append(area6)
        pros.append(area7)
        pros.append(area8)
        pros.append(area9)
        pros.append(area10)
        pros.append(area11)
        pros.append(area12)
        pros.append(area13)
        pros.append(area14)
        pros.append(area15)
        pros.append(area16)
        pros.append(area17)
        pros.append(area18)
        pros.append(area19)
        pros.append(area20)


        ids += 20

    # print(pro)
print(len(pros))
# print(pros)
pros = np.array(pros)
# ids = np.argsort(pros)
# print(ids)
f = h5py.File(
    '/home/w509/1workspace/lee/360_fix_sort/ranking_model/h5_blackdot_nums/salgan/salicon_salgan_val_multiclassi_new_0-7.h5',
    'w')
f.create_dataset('labels', data=pros)
f.close()



