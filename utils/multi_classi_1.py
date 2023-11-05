import numpy as  np
import h5py
import os
import cv2



global ids
ids = 0
train_dataset = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_rand/h5_blcakdot_nums/val_nums.h5', 'r')

# train_set = np.array(train_dataset['train'][:])
train_labels = np.array(train_dataset['labels'][:])
train_labels = train_labels.tolist()
train_label=[]
path = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/salicon_rand/vallocal/'
imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x:int(x[:-4]))
print(imgs_path)

for name in imgs_path:
    txt_path = path +name



    f = open(txt_path, "r")
    length=0
    while True:
        line = f.readline()
        if line:
            length+=1
        else:
            break



    pros = train_labels[ids: ids + length]
    aa=pros
    bb = [(aa[i - 1], i) for i in range(1, len(aa) + 1)]
    cc = sorted(bb)
    dd = [(cc[i - 1][0], i, cc[i - 1][1]) for i in range(1, length+1)]
    ee = sorted(dd, key=lambda x: x[2])
    ff = [x[1] for x in ee]
    ff_copy=[(x-1) for x in ff]
    ff_copy_copy=ff_copy.copy()
    ids += length

    if len(ff_copy)==3:
        for z in range(3):
            if ff_copy[z]==1:
                ff_copy_copy[z]+=1
            if ff_copy[z]==2:
                ff_copy_copy[z]+=2
    if len(ff_copy)==4:
        for z in range(4):
            if ff_copy[z]==1:
                ff_copy_copy[z]+=0
            if ff_copy[z]==2:
                ff_copy_copy[z]+=1
            if ff_copy[z]==3:
                ff_copy_copy[z]+=1
    # if len(ff_copy) == 5:
    #     for z in range(5):
    #         if ff_copy[z] == 1:
    #             ff_copy_copy[z] += 1
    #         if ff_copy[z] == 2:
    #             ff_copy_copy[z] += 2
    #         if ff_copy[z] == 3:
    #             ff_copy_copy[z] += 4
    #         if ff_copy[z] == 4:
    #             ff_copy_copy[z] += 5
    # if len(ff_copy) == 6:
    #     for z in range(6):
    #         if ff_copy[z] == 1:
    #             ff_copy_copy[z] += 0
    #         if ff_copy[z] == 2:
    #             ff_copy_copy[z] += 1
    #         if ff_copy[z] == 3:
    #             ff_copy_copy[z] += 2
    #         if ff_copy[z] == 4:
    #             ff_copy_copy[z] += 3
    #         if ff_copy[z] == 5:
    #             ff_copy_copy[z] += 4
    # if len(ff_copy) == 7:
    #     for z in range(7):
    #         if ff_copy[z] == 1:
    #             ff_copy_copy[z] += 0
    #         if ff_copy[z] == 2:
    #             ff_copy_copy[z] += 1
    #         if ff_copy[z] == 3:
    #             ff_copy_copy[z] += 2
    #         if ff_copy[z] == 4:
    #             ff_copy_copy[z] += 3
    #         if ff_copy[z] == 5:
    #             ff_copy_copy[z] += 3
    #         if ff_copy[z] == 6:
    #             ff_copy_copy[z] += 3
    # if len(ff_copy) == 8:
    #     for z in range(8):
    #         if ff_copy[z] == 1:
    #             ff_copy_copy[z] += 0
    #         if ff_copy[z] == 2:
    #             ff_copy_copy[z] += 0
    #         if ff_copy[z] == 3:
    #             ff_copy_copy[z] += 1
    #         if ff_copy[z] == 4:
    #             ff_copy_copy[z] += 1
    #         if ff_copy[z] == 5:
    #             ff_copy_copy[z] += 2
    #         if ff_copy[z] == 6:
    #             ff_copy_copy[z] += 2
    #         if ff_copy[z] == 7:
    #             ff_copy_copy[z] += 2
    # if len(ff_copy) == 9:
    #     for z in range(9):
    #         if ff_copy[z] == 1:
    #             ff_copy_copy[z] += 0
    #         if ff_copy[z] == 2:
    #             ff_copy_copy[z] += 0
    #         if ff_copy[z] == 3:
    #             ff_copy_copy[z] += 0
    #         if ff_copy[z] == 4:
    #             ff_copy_copy[z] += 1
    #         if ff_copy[z] == 5:
    #             ff_copy_copy[z] += 1
    #         if ff_copy[z] == 6:
    #             ff_copy_copy[z] += 1
    #         if ff_copy[z] == 7:
    #             ff_copy_copy[z] += 1
    #         if ff_copy[z] == 8:
    #             ff_copy_copy[z] += 1
    train_label.extend(ff_copy_copy)




    # for z in range(len(train_label)):
    #     if train_label[z]==1:
    #         train_label[z]+=4
    #     if train_label[z]==2:
    #         train_label[z]+=7
train_label= np.array(train_label)
print(train_label.shape)
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_rand/h5_blcakdot_nums/multiclassi_labels/val_multiclassi_0_4.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=train_label)
f.close()
