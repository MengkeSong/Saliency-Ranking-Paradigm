import os
import h5py
import math
path = r'/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/salicon_label_txt/select_train/trainlocal/'

# path1 = r'/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/boxtxt/valglobal/'
xcenter_array = []
ycenter_array = []
imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x:int(x[:-4]))
print(imgs_path)

for j in range(0,len(imgs_path)):
    print(j)
    img_txt = imgs_path[j]
    # img_nums = int(img_txt[:-4])
    # img_nums +=4000
    img_name = img_txt[:-4]
    # img_name = '{:06d}'.format(img_nums)+'.jpg'
    # gt_name = img_txt[:-8]

    # gt  = cv2.imread(gt_path)
    txt_path = path + '/' + img_txt

    f = open(txt_path, "r")
    while True:
        line = f.readline()
        if line:
            pass  # do something here
            line = line.strip()



            line = line.split(',')
            xcenter = line[4]
            ycenter = line[5]
            # print(ycenter)

            xcenter = float(xcenter)
            ycenter = float(ycenter)

            xcenter *=0.075
            ycenter *=0.075
            if xcenter==0:
                ycenter1=0
            else:
                ycenter1 = math.pow(xcenter, -1)
            if ycenter==0:
                xcenter1=0
            else:
                xcenter1 = math.pow(ycenter,-1)


            xcenter_array.append(xcenter1)
            ycenter_array.append(ycenter1)
        else:
            break
    f.close()

import numpy as np
xcenter_array= np.array(xcenter_array)
ycenter_array=np.array(ycenter_array)

f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/xycenter/xcenter_train.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('xcenter', data=xcenter_array)
f.close()

f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/xycenter/ycenter_train.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('ycenter', data=ycenter_array)
f.close()