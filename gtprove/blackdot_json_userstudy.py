import math
import os
import cv2
import json
import numpy as np
import h5py
json_file = '/media/w509/967E3C5E7E3C3977/1workspace/fix/fixations_train.json'
fixs = json.load(open(json_file,'r'))

path = r'/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/salicon_label_txt/select_train/trainlocal/'
path_img_userstudy = '/home/w509/1workspace/lee/360_fix_sort/gtprove/user_study/train3000/'

imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x:int(x[:-4]))

imgs_path_userstudy = os.listdir(path_img_userstudy)
imgs_path_userstudy.sort(key=lambda x:int(x[:-4]))
print(imgs_path)
pros = []




for j in range(0, len(imgs_path_userstudy)):

    txt_name = imgs_path[j][:-4]+'.txt'
    txt_nums = int(txt_name[:-4])
    # img_txt = imgs_path[j]
    fix = fixs[txt_nums-1]


    txt_path = path + '/' + txt_name

    f = open(txt_path, "r")
    while True:
        line = f.readline()
        if line:
            area = 0
            pass  # do something here
            line = line.strip()

            line = line.split(',')
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]





            x1 = int(float(x1))
            y1 = int(float(y1))
            x2 = int(float(x2))
            y2 = int(float(y2))

            for (y,x) in fix:
                if (x>x1 and x<x2) and (y>y1 and y<y2):
                    area+=1

            area_rate = area/len(fix)
            width = x2-x1
            height = y2 - y1

            total = width*height

            s_rate = total/(640*480)

            # area_avg = area / total
            # lam = 640*480
            alp = 0.75
            area_1 = 0*math.exp((alp*s_rate))
            area_avg = area_rate + area_1
            print(area_avg)

            # area_avg = int(area_avg)

            pros.append(area_avg)















            print("create %s {:06d}".format(j) % line)
        else:
            break
    f.close()
print(len(pros))

pros = np.array(pros)
# ids = np.argsort(pros)
# print(ids)
# f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/可视化/keshihua/train_nums.h5', 'w')
# f.create_dataset('labels', data=pros)
# f.close()
f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/user_study/user_study_3000/train_nums_0_0-0_75.h5', 'w')
f.create_dataset('labels', data=pros)
f.close()


