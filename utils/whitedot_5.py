import cv2
import numpy as np
from PIL import Image
import os
import h5py

# from zhushi import goutou
# goutou()3

area = 0
global ids
ids = 0
pros = []
path1 = '/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/2dfixdataset/box/val_salicon_fix_local/'  # 输入文件夹地址

gt_name = os.listdir(path1)
# gt_name.sort(key=lambda x:int(x[11:-4]))
gt_name.sort(key=lambda x: int(x[-10:-4]))
th_rate = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
count = 0
print(gt_name)

path = r'/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/boxtxt/vallocal/'
imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x: int(x[:-8]))
print(imgs_path)
for th in th_rate:
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
            gray_avg *= th
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


        # print(pro)
    print(len(pros))
    # print(pros)
    pros = np.array(pros)
    # ids = np.argsort(pros)
    # print(ids)
    if th>=1:
        n1 = 1
        n2 = (th-1)*10
    else:
        n1 = 0
        n2 = th*10
    # n1 = int(n1)
    # n2 = int(n2)
    f = h5py.File(
        '/home/w509/1workspace/lee/360_fix_sort/ranking_model/h5_blackdot_nums/salicon/salicon_salicon_val_multiclassi_new_{}-{}.h5'.format(n1,n2),
        'w')
    f.create_dataset('labels', data=pros)
    f.close()

    ids = 0
    pros = []



