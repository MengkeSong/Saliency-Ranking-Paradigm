import numpy as  np
import h5py
import scipy.stats as stats
from util.rmse import rmse
import warnings

warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
corr_index =0
corr_pro_index =0

train_dataset1 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/user_study/user_study_3000/multi_classi/train_multi_classi_1_0-0_75.h5', 'r')
train_dataset2 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/user_study/user_study_3000/multi_classi/train_multi_classi_0_5-0_75.h5', 'r')

train_labels1 = np.array(train_dataset1['labels'][:])
train_labels1 = train_labels1.tolist()

train_labels2 = np.array(train_dataset2['labels'][:])
train_labels2 = train_labels2.tolist()

testlen=(len(train_labels1)/5)
testlen1=(len(train_labels1)/5)

eval_srcc = 0
eval_plcc = 0
eval_rmse = 0
eval_recall = 0
eval_pre = 0
eval_f1 = 0
for i in range(0, len(train_labels1), 5):
    pros1 = train_labels1[i: i + 5]
    pros2 = train_labels2[i: i + 5]


    for cc in range(5):
        if pros1[cc]!=pros2[cc]:
            corr_pro_index+=1

    test_srcc, _ = stats.spearmanr(pros1, pros2)
    test_plcc, _ = stats.pearsonr(pros1, pros2)
    test_rmse = rmse(pros1, pros2)

    f1 = f1_score(pros2, pros1, average='macro')
    p = precision_score(pros2, pros1, average='macro')
    r = recall_score(pros2, pros1, average='macro')

    if np.isnan(test_srcc) or np.isnan(test_plcc):
        test_srcc = 0
        test_plcc = 0
        testlen -= 1
    if(test_srcc>=0.99):
        corr_index+=1


    eval_srcc += test_srcc
    eval_plcc += test_plcc
    eval_rmse += test_rmse
    eval_recall+=r
    eval_pre+=p
    eval_f1+=f1

print('=Srcc: {:.6f},Plcc: {:.6f},Rmse: {:.6f}'.format(eval_srcc / testlen, eval_plcc / testlen, eval_rmse / testlen))
print('=recall: {:.6f},precision: {:.6f},f1: {:.6f}'.format(eval_recall / testlen1, eval_pre / testlen1, eval_f1 / testlen1))
print('=corr_pro: {:.6f},corr: {:.6f}'.format(corr_pro_index / 15000, 1-(corr_index / testlen1)))