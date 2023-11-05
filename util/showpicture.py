import h5py
import numpy as np
import cv2
import os
fakenum =h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/h5_blackdot_nums/fake_nums.h5','r')
fake_nums= np.array(fakenum['labels'][:])
fake_nums = fake_nums.tolist()

val_num = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/h5_blackdot_nums/multiclassi_labels/val_multiclassi_0-5—100.h5', 'r')
val_nums= np.array(val_num['labels'][:])
val_nums = val_nums.tolist()

pred_num = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/pred_6classifier.h5', 'r')
pred_nums= np.array(pred_num['labels'][:])
pred_nums = pred_nums.tolist()

imgtxt = '/home/w509/1workspace/lee/360_fix_sort/boxtxt/dut_omron_new/vallocal/'
imgpath = '/media/w509/967E3C5E7E3C3977/1workspace/dataset/dut-omron/DUT-OMRON-image/val/'
outpath ='/media/w509/967E3C5E7E3C3977/1workspace/dataset/dut-omron/DUT-OMRON-image/shoupicture/'
imgs_path = os.listdir(imgpath)
imgs_path.sort(key=lambda x: int(x[:-4]))

imgs_txt = os.listdir(imgtxt)
imgs_txt.sort(key=lambda x: int(x[:-4]))
ids =0
ids_1 = 0
for i in range(0,len(imgs_txt)):
    val_labels = val_nums[ids_1:ids_1+6]
    pred_labels = pred_nums[ids_1:ids_1+6]
    ids_1+=6
    pros_2classifer = fake_nums[ids:ids+2]
    ids +=2
    alljpg_pro = pros_2classifer[0]
    annther_pro = pros_2classifer[1]
    imgs = cv2.imread(imgpath+imgs_path[i])
    if alljpg_pro==0:
        cv2.imwrite(outpath + imgs_path[i],imgs)
    if alljpg_pro==1:
        txt_path = imgtxt + imgs_txt[i]

        f = open(txt_path, "r")
        length = 0
        while True:
            line = f.readline()
            if line:
                length += 1
            else:
                break
        length -=annther_pro
        f = open(txt_path, "r")
        for t in range(0,length):
            line = f.readline()
            if line:
                line = line.split(',')
                x1 = line[0]
                x1 = int(x1)
                y1 = line[1]
                y1 = int(y1)
                x2 = line[2]
                x2 = int(x2)
                y2 = line[3]
                y2 = int(y2)
                if pred_labels[t]==val_labels[t]:
                    imgs = cv2.rectangle(imgs,(x1,y1),(x2,y2),(0,255,0),4)
                else:
                    imgs = cv2.rectangle(imgs,(x1,y1),(x2,y2),(255,0,0),4)

            else:
                break
        cv2.imwrite(outpath+imgs_path[i],imgs)





