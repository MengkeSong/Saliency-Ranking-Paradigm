import os
import cv2
import json
import numpy as np
import h5py
json_file = '/media/w509/967E3C5E7E3C3977/1workspace/dataset/dut-omron/json/fixation_val.json'
fixs = json.load(open(json_file,'r'))

path = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/dut-omron/vallocal/'

imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x:int(x[:-4]))
print(imgs_path)
pros = []




for j in range(0, len(imgs_path)):


    img_txt = imgs_path[j]
    fix = fixs[j]

    txt_path = path + '/' + img_txt

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

            for (y,x,_) in fix:
                if (x>x1 and x<x2) and (y>y1 and y<y2):
                    area+=1


            width = x2-x1
            height = y2 - y1

            total = width*height

            # area_avg = area / total
            area_avg = area
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
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron/h5_blackdot_nums/val_nums.h5', 'w')
f.create_dataset('labels', data=pros)
f.close()


