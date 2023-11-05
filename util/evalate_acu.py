import numpy as  np
import h5py
import scipy.stats as stats
from util.rmse import rmse
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

train_dataset1 = h5py.File('/home/w509/1workspace/lee/ranking_model/h5_blackdot_nums/multiclassi_labels/salicon_salfbnet_val_multiclassi_new_1_5.h5', 'r')
train_dataset2 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/可视化/keshihua/val_multiclassi.h5', 'r')

train_labels1 = np.array(train_dataset1['labels'][:])
train_labels1 = train_labels1.tolist()

train_labels2 = np.array(train_dataset2['labels'][:])
train_labels2 = train_labels2.tolist()

# testlen=846
# testlen1=846
global ids
ids = 0
eval_srcc = 0
eval_plcc = 0
eval_rmse = 0
eval_recall = 0
eval_pre = 0
eval_f1 = 0

nums_first = 0
nums_second = 0
nums_third = 0

testlen_first = 0
testlen_second = 0
testlen_third = 0



def isnull(list):
    gt_id = 0
    for gt in list:
        if gt > 0:
            gt_id = 1
    return gt_id
for label in range(0,len(train_labels1),5):
    length=5


    pros_gt = train_labels1[ids: ids + length]
    pros2 = train_labels2[ids: ids + length]
    ids += length



    testlen_first+=1
    index_gt1 = pros_gt.index(max(pros_gt))
    if pros_gt[index_gt1]==pros2[index_gt1]:
        nums_first+=1
    del pros_gt[index_gt1]
    del pros2[index_gt1]

    testlen_second+=1
    index_gt2 = pros_gt.index(max(pros_gt))
    if pros_gt[index_gt2]==pros2[index_gt2]:
        nums_second+=1
    del pros_gt[index_gt2]
    del pros2[index_gt2]

    testlen_third+=1
    index_gt3 = pros_gt.index(max(pros_gt))
    if pros_gt[index_gt3]==pros2[index_gt3]:
        nums_third+=1
    del pros_gt[index_gt3]
    del pros2[index_gt3]
acc_first = nums_first/testlen_first
acc_second = nums_second/testlen_second
acc_third = nums_third/testlen_third
print('ACC_first:{:04f},ACC_second:{:04f},ACC_third:{:04f}'.format(acc_first,acc_second,acc_third))










