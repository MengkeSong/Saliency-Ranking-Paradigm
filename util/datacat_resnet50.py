import h5py
import numpy as np

train_dataset0 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon/res50val_local_global_unisal.h5', 'r')
# train_dataset1 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_resv2/resv2val_global.h5','r')
labels_dataset = h5py.File('/home/w509/1workspace/lee/360_fix_sort/可视化/keshihua/val_multiclassi.h5', 'r')
train_set0 = np.array(train_dataset0['train'][:])
train_set = train_set0[:,0:4096]
# train_set1 = np.array(train_dataset1['train'][:])
labels_set = np.array(labels_dataset['labels'][:])
# train_set = np.concatenate((train_set0,train_set1),1)
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon/train_val_ture/val_local_global_multi_classi_fixs_true.h5', 'w')
f.create_dataset('train', data=train_set)


f.create_dataset('labels', data=labels_set)
# f.create_dataset('xcenter', data=train_set2)
# f.create_dataset('ycenter', data=train_set3)

f.close()