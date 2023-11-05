import os
from shutil import copy
import h5py
import numpy as np
nums = 250


gt_labels_path = '/gtprove/anothergt/fixation-area/val_multi_classi.h5'
select_gts = []
train_dataset = h5py.File(gt_labels_path, 'r')

# train_set = np.array(train_dataset['train'][:])
gt_labels=np.array(train_dataset['labels'][:])
imgtxt = '/home/w509/1workspace/lee/360_fix_sort/gtprove/selecttxt/'
txts_path = os.listdir(imgtxt)
txts_path.sort(key=lambda x:int(x[:-8]))
print(txts_path)
# imgs_txt = os.listdir(imgtxt)
# imgs_txt.sort(key=lambda x:int(x[:-8]))
# print(imgs_txt)
tocopyimgpath = '/home/w509/1workspace/lee/360_fix_sort/gtprove/selectpic/'
tocopytxtpath = '/home/w509/1workspace/lee/360_fix_sort/gtprove/selecttxt/'
for num in range(0,nums):
    txt_path = txts_path[num]
    txt_nums = int(txt_path[:-8])
    gt_inter = gt_labels[(txt_nums-1)*5:(txt_nums)*5]
    select_gts.extend(gt_inter)

select_gts = np.array(select_gts)
print(select_gts.shape)
f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/selectgt/select_another_gts_labels.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=select_gts)
f.close()
