import h5py
import numpy as np

test_dataset0 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/unisal/cut_feature_true/val_unisal_roi_local.h5', 'r')
test_dataset1 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/unisal/cut_feature_true/val_unisal_roi_global.h5','r')
# train_dataset2 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_resv2/resv2train_sal_unisal.h5','r')
test_dataset3 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/xycenter/xcenter_val.h5', 'r')
test_dataset4 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/xycenter/ycenter_val.h5', 'r')
# train_dataset5 = h5py.File('/home/w509/1workspace/lee/2dfix_classi/feature_salcon/xycenter_h5/train_size.h5', 'r')
test_labels_dataset = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/val_multi_classi_0_5-0_75.h5', 'r')

train_dataset0 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/unisal/cut_feature_true/train_unisal_roi_local.h5', 'r')
train_dataset1 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/unisal/cut_feature_true/train_unisal_roi_global.h5','r')
# train_dataset2 = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_resv2/resv2train_sal_unisal.h5','r')
train_dataset3 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/xycenter/xcenter_train.h5', 'r')
train_dataset4 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/xycenter/ycenter_train.h5', 'r')
# train_dataset5 = h5py.File('/home/w509/1workspace/lee/2dfix_classi/feature_salcon/xycenter_h5/train_size.h5', 'r')
labels_dataset = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/train_multi_classi_0_5-0_75.h5', 'r')


train_set0 = np.array(train_dataset0['train'][:])
train_set1 = np.array(train_dataset1['train'][:])
# train_set2 = np.array(train_dataset2['train'][:])
train_set3 = np.array(train_dataset3['xcenter'][:])
train_set4 = np.array(train_dataset4['ycenter'][:])
# train_set5 = np.array(train_dataset5['labels'][:])
labels_set = np.array(labels_dataset['labels'][:])

train_set3 = np.expand_dims(train_set3,1)
train_set4 = np.expand_dims(train_set4,1)

test_set0 = np.array(test_dataset0['train'][:])
# test_set0 = test_set0[0:3000]
test_set1 = np.array(test_dataset1['train'][:])
# test_set1 = test_set1[0:3000]
# train_set2 = np.array(train_dataset2['train'][:])
test_set3 = np.array(test_dataset3['xcenter'][:])
# test_set3 = test_set3[0:3000]
test_set4 = np.array(test_dataset4['ycenter'][:])
# test_set4 = test_set4[0:3000]

test_labels_set = np.array(test_labels_dataset['labels'][:])
# test_labels_set=test_labels_set[0:3000]
test_set3 = np.expand_dims(test_set3,1)
test_set4 = np.expand_dims(test_set4,1)

# train_set0 = np.concatenate((train_set0,test_set0),axis=0)
# train_set1 = np.concatenate((train_set1,test_set1),axis=0)
# train_set3 = np.concatenate((train_set3,test_set3),axis=0)
# train_set4 = np.concatenate((train_set4,test_set4),axis=0)
# labels_set = np.concatenate((labels_set,test_labels_set),axis=0)
# train_set2 = np.array(train_dataset2['train'][:])
# train_set = np.concatenate((train_set0,train_set1,train_set2),axis=1)
# print(train_set.shape)
# train_labels=np.array(train_dataset['labels'][:])


f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/feature/ranking_feature/train_local_global_unisal_multi_classi_0_5-0_75.h5', 'w')
f.create_dataset('train1', data=train_set0)
f.create_dataset('train2', data=train_set1)
f.create_dataset('xcenter', data=train_set3)
f.create_dataset('ycenter', data=train_set4)
# f.create_dataset('train1', data=train_set0)

f.create_dataset('labels', data=labels_set)
# f.create_dataset('xcenter', data=train_set2)
# f.create_dataset('ycenter', data=train_set3)

f.close()
f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/feature/ranking_feature/val_local_global_unisal_multi_classi_0_5-0_75.h5', 'w')
f.create_dataset('train1', data=test_set0)
f.create_dataset('train2', data=test_set1)
f.create_dataset('xcenter', data=test_set3)
f.create_dataset('ycenter', data=test_set4)
# f.create_dataset('train1', data=train_set0)

f.create_dataset('labels', data=test_labels_set)
# f.create_dataset('xcenter', data=train_set2)
# f.create_dataset('ycenter', data=train_set3)

f.close()
