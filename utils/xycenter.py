import os
import cv2

# davis_train
# davis_test
# Visal
# Segtrack-v2
# Easy-35
# VOS_test
global image_id
image_id = 0
global image_id1
image_id1 = 0
xcenter_array = []
ycenter_array = []

path = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/dut_omron_new/vallocal/'
path1 = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/dut_omron_new/valglobal/'

imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x:int(x[:-4]))
print(imgs_path)

imgs_path1 = os.listdir(path1)
imgs_path1.sort(key=lambda x:int(x[:-4]))



for j in range(0, len(imgs_path)):


    img_txt = imgs_path[j]
    img_name = img_txt[:-4] + '.jpg'
    gt_name = img_txt[:-4]
    img_path = r'/media/w509/967E3C5E7E3C3977/1workspace/dataset/dut-omron/DUT-OMRON-image/val/' + '%s' % (img_name)
    img = cv2.imread(img_path)
    img_width = img.shape[1]
    resize_width = img_width/7
    img_height = img.shape[0]
    resize_height = img_height/7
    txt_path = path + '/' + img_txt

    f = open(txt_path, "r")
    while True:
        line = f.readline()
        if line:
            pass  # do something here
            line = line.strip()



            image_id +=1



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
            xcenter = int(float(xcenter))
            xcenter = xcenter/resize_width
            # xcenter = int(xcenter)

            ycenter = int(float(ycenter))
            ycenter = ycenter/resize_height
            # ycenter = int(ycenter)


            xcenter_array.append(xcenter)
            ycenter_array.append(ycenter)
            xycenter_path = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/xycenter'

            if not os.path.exists(xycenter_path):
                os.makedirs(xycenter_path)

            print("create %s {:06d}".format(j) % line)
        else:
            xcenter = 3.5
            # xcenter = int(xcenter)
            ycenter = 3.5
            # ycenter = int(ycenter)
            xcenter_array.append(xcenter)
            ycenter_array.append(ycenter)

            break
    f.close()

import numpy as np

xcenter_array=np.array(xcenter_array)
ycenter_array=np.array(ycenter_array)

print(xcenter_array.shape)
print(ycenter_array.shape)
import h5py
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/xycenter/xcenter_val_adjpg.h5', 'w')
f.create_dataset('xcenter', data=xcenter_array)
f.close()
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/xycenter/ycenter_val_adjpg.h5', 'w')
f.create_dataset('ycenter', data=ycenter_array)
f.close()


