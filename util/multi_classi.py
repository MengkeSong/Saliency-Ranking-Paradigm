import numpy as  np
import h5py
import os
import cv2



global ids
ids = 0
train_dataset = h5py.File('/home/w509/1workspace/lee/ATSal-master/my360output_box/unisal_trainnums_new_0-5.h5', 'r')

# train_set = np.array(train_dataset['train'][:])
train_labels = np.array(train_dataset['labels'][:])
train_labels = train_labels.tolist()
train_label=[]
path = r'/home/w509/1workspace/360biaozhu/val_txts/'
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


    train_label.extend(ff_copy_copy)


train_label= np.array(train_label)
print(train_label.shape)
f = h5py.File('/home/w509/1workspace/lee/ATSal-master/my360output_box/atsal_train.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=train_label)
f.close()
