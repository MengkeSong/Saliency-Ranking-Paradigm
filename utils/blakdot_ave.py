import cv2
import numpy as np
from PIL import Image
import os
import h5py

import torch
txtpath =r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/360_new_dataset/test/testlocal/'

# boxpath = os.listdir(txtpath)
# boxpath.sort(key=lambda x:int(x[:-8]))
path = '/media/w509/967E3C5E7E3C3977/1workspace/360_new_dataset/gt/test/'
gtdir = os.listdir(path)
gtdir.sort(key=lambda x:int(x[:]))
global ids
ids =1
pros = []
for dir in gtdir:
    dir_name=int(dir)
    if dir_name<400:

        gt = os.listdir(path+dir)
        gt.sort(key=lambda x:int(x[:-4]))
        for i in range(0,len(gt)):
            gt_png = gt[i]
            gt_name = gt_png[1:-4]
            gt_name = int(gt_name)

            txt = txtpath + '{:06d}'.format(ids) + '.txt'
            if gt_name % 20 ==0 and (gt_name!=300 and gt_name!=200):
                ids+=1
                gt_1 = cv2.imread(path + dir + '/' + gt[i - 1])
                gt_2 = cv2.imread(path + dir + '/' + gt[i])
                print('目录'+dir)
                gt_3 = cv2.imread(path + dir + '/' + gt[i + 1])
                file = open(txt, "r")
                while True:
                    line = file.readline()
                    if line:
                        pass  # do something here
                        line = line.strip()



                        line = line.split(',')
                        x1 = line[0]
                        y1 = line[1]
                        x2 = line[2]
                        y2 = line[3]
                        xcenter = line[4]
                        ycenter = line[5]
                        # print(ycenter)

                        x1 = int(float(x1))
                        y1 = int(float(y1))
                        x2 = int(float(x2))
                        y2 = int(float(y2))
                        gt_box1 = gt_1[y1:y2, x1:x2]
                        gt_box2 = gt_2[y1:y2, x1:x2]
                        gt_box3 = gt_3[y1:y2, x1:x2]

                        gray_1 = cv2.cvtColor(gt_box1, cv2.COLOR_BGR2GRAY)

                        gray_1_nums = gray_1.sum()
                        # retval_1, dst_1 = cv2.threshold(gray_1, 23, 255, cv2.THRESH_BINARY)
                        gray_2 = cv2.cvtColor(gt_box2, cv2.COLOR_BGR2GRAY)

                        gray_2_nums = gray_2.sum()
                        # retval_2, dst_2 = cv2.threshold(gray_2, 23, 255, cv2.THRESH_BINARY)

                        gray_3 = cv2.cvtColor(gt_box3, cv2.COLOR_BGR2GRAY)

                        gray_3_nums = gray_3.sum()
                        # retval_3, dst_3 = cv2.threshold(gray_3, 23, 255, cv2.THRESH_BINARY)

                        height, width = gray_1.shape

                        total = height*width

                        gray_1_avg = gray_1_nums / total
                        gray_2_avg = gray_2_nums / total
                        gray_3_avg = gray_3_nums / total




                        # area_1 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_1[t, f] == 255:
                        #             area_1 += 1
                        # area_2 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_2[t, f] == 255:
                        #             area_2 += 1
                        # area_3 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_3[t, f] == 255:
                        #             area_3 += 1
                        # area = (0.8*area_1+area_2+0.8*area_3)/3
                        # area = int(area)

                        gray_avg = (0.8*gray_1_avg + gray_2_avg +0.8*gray_3_avg)/3
                        print(gray_avg)
                        pros.append(gray_avg)
                    else:
                        break
                file.close()
            elif gt_name %20 ==0 and (gt_name==300 or gt_name==200):
                ids += 1
                gt_1 = cv2.imread(path + dir + '/' + gt[i - 1])
                gt_2 = cv2.imread(path + dir + '/' + gt[i])

                file = open(txt, "r")
                while True:
                    line = file.readline()
                    if line:
                        pass  # do something here
                        line = line.strip()

                        line = line.split(',')
                        x1 = line[0]
                        y1 = line[1]
                        x2 = line[2]
                        y2 = line[3]
                        xcenter = line[4]
                        ycenter = line[5]
                        # print(ycenter)

                        x1 = int(float(x1))
                        y1 = int(float(y1))
                        x2 = int(float(x2))
                        y2 = int(float(y2))
                        gt_box1 = gt_1[y1:y2, x1:x2]
                        gt_box2 = gt_2[y1:y2, x1:x2]


                        gray_1 = cv2.cvtColor(gt_box1, cv2.COLOR_BGR2GRAY)
                        gray_1_nums = gray_1.sum()
                        # retval_1, dst_1 = cv2.threshold(gray_1, 23, 255, cv2.THRESH_BINARY)
                        gray_2 = cv2.cvtColor(gt_box2, cv2.COLOR_BGR2GRAY)
                        gray_2_nums = gray_2.sum()
                        # retval_2, dst_2 = cv2.threshold(gray_2, 23, 255, cv2.THRESH_BINARY)



                        height, width = gray_1.shape
                        total = height * width

                        gray_1_avg = gray_1_nums / total
                        gray_2_avg = gray_2_nums / total
                        # area_1 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_1[t, f] == 255:
                        #             area_1 += 1
                        # area_2 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_2[t, f] == 255:
                        #             area_2 += 1
                        #
                        # area = (0.8 * area_1 + area_2 ) / 2
                        # area = int(area)

                        gray_avg = (0.8*gray_1_avg +gray_2_avg)/2
                        print(gray_avg)
                        pros.append(gray_avg)
                    else:
                        break
                file.close()
    else:
        gt = os.listdir(path + dir)
        gt.sort(key=lambda x: int(x[:-4]))
        for i in range(0, len(gt)):
            gt_png = gt[i]
            gt_name = gt_png[1:-4]
            gt_name = int(gt_name)

            txt = txtpath + '{:06d}'.format(ids) + '.txt'
            if gt_name % 20 == 0 and (gt_name != 300 and gt_name!=200):
                ids += 1
                gt_1 = cv2.imread(path + dir + '/' + gt[i - 1])
                gt_2 = cv2.imread(path + dir + '/' + gt[i])
                gt_3 = cv2.imread(path + dir + '/' + gt[i + 1])
                file = open(txt, "r")
                while True:
                    line = file.readline()
                    if line:
                        pass  # do something here
                        line = line.strip()

                        line = line.split(',')
                        x1 = line[0]
                        y1 = line[1]
                        x2 = line[2]
                        y2 = line[3]
                        xcenter = line[4]
                        ycenter = line[5]
                        # print(ycenter)

                        x1 = int(float(x1))
                        y1 = int(float(y1))
                        x2 = int(float(x2))
                        y2 = int(float(y2))
                        gt_box1 = gt_1[y1:y2, x1:x2]
                        gt_box2 = gt_2[y1:y2, x1:x2]
                        gt_box3 = gt_3[y1:y2, x1:x2]

                        gray_1 = cv2.cvtColor(gt_box1, cv2.COLOR_BGR2GRAY)
                        gray_1_nums = gray_1.sum()
                        # retval_1, dst_1 = cv2.threshold(gray_1, 30, 255, cv2.THRESH_BINARY)
                        gray_2 = cv2.cvtColor(gt_box2, cv2.COLOR_BGR2GRAY)
                        gray_2_nums = gray_2.sum()
                        # retval_2, dst_2 = cv2.threshold(gray_2, 30, 255, cv2.THRESH_BINARY)

                        gray_3 = cv2.cvtColor(gt_box3, cv2.COLOR_BGR2GRAY)
                        gray_3_nums = gray_3.sum()
                        # retval_3, dst_3 = cv2.threshold(gray_3, 30, 255, cv2.THRESH_BINARY)

                        height, width = gray_1.shape
                        total = height * width

                        gray_1_avg = gray_1_nums / total
                        gray_2_avg = gray_2_nums / total
                        gray_3_avg = gray_3_nums / total

                        # area_1 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_1[t, f] == 255:
                        #             area_1 += 1
                        # area_2 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_2[t, f] == 255:
                        #             area_2 += 1
                        # area_3 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_3[t, f] == 255:
                        #             area_3 += 1
                        # area = (0.8 * area_1 + area_2 + 0.8 * area_3) / 3

                        gray_avg = (0.8 * gray_1_avg + gray_2_avg + 0.8 * gray_3_avg) / 3
                        print(gray_avg)
                        pros.append(gray_avg)
                    else:
                        break
                file.close()
            elif gt_name % 20 == 0 and (gt_name == 300 or gt_name==200):
                ids += 1
                gt_1 = cv2.imread(path + dir + '/' + gt[i - 1])
                gt_2 = cv2.imread(path + dir + '/' + gt[i])

                file = open(txt, "r")
                while True:
                    line = file.readline()
                    if line:
                        pass  # do something here
                        line = line.strip()

                        line = line.split(',')
                        x1 = line[0]
                        y1 = line[1]
                        x2 = line[2]
                        y2 = line[3]
                        xcenter = line[4]
                        ycenter = line[5]
                        # print(ycenter)

                        x1 = int(float(x1))
                        y1 = int(float(y1))
                        x2 = int(float(x2))
                        y2 = int(float(y2))
                        gt_box1 = gt_1[y1:y2, x1:x2]
                        gt_box2 = gt_2[y1:y2, x1:x2]

                        gray_1 = cv2.cvtColor(gt_box1, cv2.COLOR_BGR2GRAY)
                        gray_1_nums =gray_1.sum()
                        # retval_1, dst_1 = cv2.threshold(gray_1, 30, 255, cv2.THRESH_BINARY)
                        gray_2 = cv2.cvtColor(gt_box2, cv2.COLOR_BGR2GRAY)
                        gray_2_nums = gray_2.sum()
                        # retval_2, dst_2 = cv2.threshold(gray_2, 30, 255, cv2.THRESH_BINARY)

                        height, width = gray_1.shape
                        total = height * width

                        gray_1_avg = gray_1_nums / total
                        gray_2_avg = gray_2_nums / total

                        # area_1 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_1[t, f] == 255:
                        #             area_1 += 1
                        # area_2 = 0
                        # for t in range(height):
                        #     for f in range(width):
                        #         if dst_2[t, f] == 255:
                        #             area_2 += 1
                        #
                        # area = (0.8 * area_1 + area_2) / 2
                        gray_avg = (0.8 * gray_1_avg + gray_2_avg)/2
                        print(gray_avg)
                        pros.append(gray_avg)
                    else:
                        break
                file.close()


print(len(pros))
print(ids)
f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/feature/360_new_dataset/h5_blackdot_nums/test_360nums_gray_avg.h5', 'w')
f.create_dataset('labels', data=pros)
f.close()





