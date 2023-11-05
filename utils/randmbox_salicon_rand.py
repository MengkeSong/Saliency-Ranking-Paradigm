import os
import cv2
from cut_panoimg import cutOutPannoama
from PIL import Image
import h5py
import numpy as np

global image_id
image_id = 0
global image_id1
image_id1 = 0
xcenter_array = []
ycenter_array = []


pathtxt1 = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/salicon_rand/vallocal/'
pathtxt2 = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/salicon_rand/valglobal/'

txtpath1 = os.listdir(pathtxt1)
txtpath2 = os.listdir(pathtxt2)
txtpath1.sort(key=lambda x:str(x[:-4]))
txtpath2.sort(key=lambda x:str(x[:-4]))




# imgs_path3 = os.listdir(path3)
# imgs_path3.sort(key=lambda x:str(x[:-4]))

# imgs_path2 = os.listdir(path2)
# imgs_path2.sort(key=lambda x:str(x[:-4]))

# if not os.path.exists(img_flo):
#     os.makedirs(img_flo)
# if not os.path.exists(img_gt):
#     os.makedirs(img_gt)
global ids
ids = 1
global ids_global
ids_global = 1

global ids_gt
ids_gt = 1
j=0



gt_dataset = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_multi_rand/salicon_rand/h5_blackdot_nums/val_nums.h5', 'r')
gt_dataset = np.array(gt_dataset['labels'][:])

gt_set = []


for i in range(0,len(txtpath1)):
    img_txt = txtpath1[i]
    print(img_txt)
    txt_path = pathtxt1 + '/' + img_txt
    f = open(txt_path, "r")
    prolen = 0
    while True:

        line = f.readline()
        if line:
            prolen+=1
        else:
            break
    f.close()

    # img_f = imgs_path3[j:j+prolen]
    gt_labels = gt_dataset [j:j+prolen]
    j+=prolen
    # img_t = imgs_path2[j:j+10]


    if prolen==5:
        for t in range(0, 3):

            gt_set.append(gt_labels[t])


            ids += 1
        for t in range(1, 4):

            gt_set.append(gt_labels[t])



            ids += 1
        for t in range(2, 5):
            gt_set.append(gt_labels[t])


            ids += 1
        for t in [3,4,0]:

            gt_set.append(gt_labels[t])

            ids += 1
        for t in [4,0,1]:
            gt_set.append(gt_labels[t])

            ids += 1
    if prolen==4:
        for t in range(0,3):

            gt_set.append(gt_labels[t])

            ids += 1
        for t in range(1,4):

            gt_set.append(gt_labels[t])

            ids += 1
        for t in [2,3,0]:
            gt_set.append(gt_labels[t])

            ids += 1
        for t in [3,0,1]:

            gt_set.append(gt_labels[t])

            ids += 1
    if prolen==3:
        for t in range(0,3):

            gt_set.append(gt_labels[t])


            ids += 1






gt_set_np=np.array(gt_set)
print(gt_set_np.shape)
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_multi_rand/h5_blackdot_nums/val_nums.h5', 'w')

f.create_dataset('labels', data=gt_set_np)
f.close()












