import h5py
import numpy as np

train_dataset0 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/unisal_clone/feature_pascal-s_unisal/val_unisal_roi_local.h5', 'r')
train_dataset1 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/unisal_clone/feature_pascal-s_unisal/val_unisal_roi_global.h5','r')
# train_dataset2 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_resv2/resv2train_sal_unisal.h5','r')
# train_dataset3 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/my360dataset/xycenter/val_xcenter.h5', 'r')
# train_dataset4 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/my360dataset/xycenter/val_ycenter.h5', 'r')
# train_dataset5 = h5py.File('/home/w509/1workspace/lee/2dfix_classi/feature_salcon/xycenter_h5/train_size.h5', 'r')
labels_dataset = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/pascal-s/h5_blackdot_nums/multiclassi_labels/unisal_val_multiclassi.h5', 'r')
train_set0 = np.array(train_dataset0['train'][:])
train_set1 = np.array(train_dataset1['train'][:])
# train_set2 = np.array(train_dataset2['train'][:])
# train_set3 = np.array(train_dataset3['xcenter'][:])
# train_set4 = np.array(train_dataset4['ycenter'][:])
# train_set5 = np.array(train_dataset5['labels'][:])
labels_set = np.array(labels_dataset['labels'][:])

# train_set3 = np.expand_dims(train_set3,1)
# train_set4 = np.expand_dims(train_set4,1)
# train_set5 = np.expand_dims(train_set5,1)


# train_set2 = np.array(train_dataset2['train'][:])
# train_set = np.concatenate((train_set0,train_set1,train_set2),axis=1)
# print(train_set.shape)
# train_labels=np.array(train_dataset['labels'][:])


f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/pascal-s/train_val/val_local_global_unisal_multi_classi_fixs.h5', 'w')
f.create_dataset('train1', data=train_set0)
f.create_dataset('train2', data=train_set1)
# f.create_dataset('xcenter', data=train_set3)
# f.create_dataset('ycenter', data=train_set4)
# f.create_dataset('train1', data=train_set0)

f.create_dataset('labels', data=labels_set)
# f.create_dataset('xcenter', data=train_set2)
# f.create_dataset('ycenter', data=train_set3)

f.close()

# filepath='/home/by512/1workspace/lee/pytorch_ExtractFeature-master/references/classification/dh5/data1train.h5'
# h5File = h5py.File(filepath, 'r')
# label=h5File['labels']
# data = h5File['train']

"""AVE dataset"""
import numpy as np
import torch
import h5py

# f = h5py.File('./datah5/data11test.h5', 'w')
#         # namess = []
#         # for j in labels:
#         #     namess.append(j.encode())
#         # labels_np = np.array(namess)
#
# f.create_dataset('test11', data=test_set3)
# f.create_dataset('labels11', data=test_labels)
# # f.create_dataset('X_test', data=Test_image)
# # f.create_dataset('y_test', data=Test_label)
# f.close()
# class DataFromH5File(data.Dataset):
#     def __init__(self, filepath):
#         h5File = h5py.File(filepath, 'r')
#         self.data = h5File['train']
#         self.labels = h5File['labels']
#
#     def __getitem__(self, idx):
#         label = torch.from_numpy(self.labels[idx]).float()
#         data = torch.from_numpy(self.data[idx]).float()
#         return data, label
#
#     def __len__(self):
#         assert self.data.shape[0] == self.labels.shape[0], "Wrong data length"
#         return self.data.shape[0]