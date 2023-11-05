import json
import h5py
import numpy as np
path1 = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/my360dataset/fixation_val_zhu.json'
list_all =[]
all_label_list=json.load(open(path1,'r'))
for list1 in all_label_list:
    list_all.extend(list1)

list_all=np.array(list_all)
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/my360dataset/fixation_val_zhu.h5', 'w')
f.create_dataset('labels', data=list_all)
f.close()