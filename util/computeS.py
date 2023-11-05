import cv2
import os
import h5py

txt_path ='/home/w509/1workspace/360biaozhu/val_txts/'
h5_path ='/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/my360dataset/h5_blackdot_nums/val_nums.h5'
txts_path = os.listdir(txt_path)

txts_path.sort(key=lambda x:int(x[:-4]))
id = 0
gt_list = []
for txt_name in txts_path:
    txt = txt_path + txt_name


    f = open(txt, "r")
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

            s = (x2-x1)*(y2-y1)

            gt_list.append(s)
            id+=1
            print(id)

        else:
            break
    f.close()
import numpy as np
gt_list = np.array(gt_list)

f = h5py.File(h5_path, 'w')
f.create_dataset('labels', data=gt_list)


f.close()
