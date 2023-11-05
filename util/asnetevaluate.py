# import numpy as  np
# import h5py
# import scipy.stats as stats
# from util.rmse import rmse
# import warnings
#
# warnings.filterwarnings('ignore')
# from sklearn import metrics
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
#
# srcc_list = []
# plcc_list = []
# for t1 in range(0,2):
#     if t1 ==0:
#         for t2 in range(0,10):
#
#
#             train_dataset1 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/ranking_model/h5_blackdot_nums/multiclassi_labels/deepgaze2e/salicon_deepgaze2e_val_multiclassi_new_{}-{}.h5'.format(t1,t2), 'r')
#             train_dataset2 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/fixation-lam-alp-area/val_multi_classi_640480-300.h5', 'r')
#
#             train_labels1 = np.array(train_dataset1['labels'][:])
#             train_labels1 = train_labels1.tolist()
#
#             train_labels2 = np.array(train_dataset2['labels'][:])
#             train_labels2 = train_labels2.tolist()
#
#             testlen=(len(train_labels1)/5)
#             testlen1=(len(train_labels1)/5)
#
#             eval_srcc = 0
#             eval_plcc = 0
#             eval_rmse = 0
#             eval_recall = 0
#             eval_pre = 0
#             eval_f1 = 0
#             for i in range(0, len(train_labels1), 5):
#                 pros1 = train_labels1[i: i + 5]
#                 pros2 = train_labels2[i: i + 5]
#
#                 test_srcc, _ = stats.spearmanr(pros1, pros2)
#                 test_plcc, _ = stats.pearsonr(pros1, pros2)
#                 test_rmse = rmse(pros1, pros2)
#
#                 f1 = f1_score(pros2, pros1, average='macro')
#                 p = precision_score(pros2, pros1, average='macro')
#                 r = recall_score(pros2, pros1, average='macro')
#
#                 if np.isnan(test_srcc) or np.isnan(test_plcc):
#                     test_srcc = 0
#                     test_plcc = 0
#                     testlen -= 1
#
#                 eval_srcc += test_srcc
#                 eval_plcc += test_plcc
#                 eval_rmse += test_rmse
#                 eval_recall+=r
#                 eval_pre+=p
#                 eval_f1+=f1
#             srcc_list.append(eval_srcc / testlen)
#             print('=Srcc: {:.6f},Plcc: {:.6f},Rmse: {:.6f}'.format(eval_srcc / testlen, eval_plcc / testlen, eval_rmse / testlen))
#             print('=recall: {:.6f},precision: {:.6f},f1: {:.6f}'.format(eval_recall / testlen1, eval_pre / testlen1, eval_f1 / testlen1))
#     else:
#         for t2 in range(0, 6):
#
#             train_dataset1 = h5py.File(
#                 '/home/w509/1workspace/lee/360_fix_sort/ranking_model/h5_blackdot_nums/multiclassi_labels/deepgaze2e/salicon_deepgaze2e_val_multiclassi_new_{}-{}.h5'.format(
#                     t1, t2), 'r')
#             train_dataset2 = h5py.File(
#                 '/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/fixation-lam-alp-area/val_multi_classi_640480-300.h5',
#                 'r')
#
#             train_labels1 = np.array(train_dataset1['labels'][:])
#             train_labels1 = train_labels1.tolist()
#
#             train_labels2 = np.array(train_dataset2['labels'][:])
#             train_labels2 = train_labels2.tolist()
#
#             testlen = (len(train_labels1) / 5)
#             testlen1 = (len(train_labels1) / 5)
#
#             eval_srcc = 0
#             eval_plcc = 0
#             eval_rmse = 0
#             eval_recall = 0
#             eval_pre = 0
#             eval_f1 = 0
#             for i in range(0, len(train_labels1), 5):
#                 pros1 = train_labels1[i: i + 5]
#                 pros2 = train_labels2[i: i + 5]
#
#                 test_srcc, _ = stats.spearmanr(pros1, pros2)
#                 test_plcc, _ = stats.pearsonr(pros1, pros2)
#                 test_rmse = rmse(pros1, pros2)
#
#                 f1 = f1_score(pros2, pros1, average='macro')
#                 p = precision_score(pros2, pros1, average='macro')
#                 r = recall_score(pros2, pros1, average='macro')
#
#                 if np.isnan(test_srcc) or np.isnan(test_plcc):
#                     test_srcc = 0
#                     test_plcc = 0
#                     testlen -= 1
#
#                 eval_srcc += test_srcc
#                 eval_plcc += test_plcc
#                 eval_rmse += test_rmse
#                 eval_recall += r
#                 eval_pre += p
#                 eval_f1 += f1
#             srcc_list.append(eval_srcc / testlen)
#             print('=Srcc: {:.6f},Plcc: {:.6f},Rmse: {:.6f}'.format(eval_srcc / testlen, eval_plcc / testlen,
#                                                                    eval_rmse / testlen))
#             print('=recall: {:.6f},precision: {:.6f},f1: {:.6f}'.format(eval_recall / testlen1, eval_pre / testlen1,
#                                                                         eval_f1 / testlen1))
#
# import matplotlib.pyplot as plt
#
# names = ['0', '0.1', '0.2', '0.3', '0.4','0.5','0.6','0.7','0.8','0.9','1','1.1','1.2','1.3','1.4','1.5']
# x = range(len(names))
# y1 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
# y2 = [0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789,0.789]
# # y = [0.9595,0.9303,0.9042,0.8794,0.8521,0.8255,0.8013,0.7733,0.7467,0.7194,0.6949,0.6729,0.6511,0.6305,0.6111,0.5941]
# print(srcc_list)
# def f_mea(a,b):
#     fmeasure = (2*a*b)/(a+b)
#     return fmeasure
# z = []
# for i in range(len(srcc_list)):
#     f = f_mea(y1[i],srcc_list[i])
#     z.append(f)
# print(z)
#
# # y = [0.2786,0.09526,0.03631,0.01232,0.00291]
#
# # y1=[0.452,0.1272,0.03737902,0.01120912,0.00281906]
# plt.plot(x, srcc_list, marker='o', mec='r', mfc='w',label=u'th-srcc')
# plt.plot(x, z, marker='*', ms=10,label=u'fmeasure')
# plt.plot(x, y2, marker='*', ms=10,label=u'mymethod')
#
# plt.legend()  # 让图例生效
# plt.xticks(x, names, rotation=45)
# plt.margins(0)
# plt.subplots_adjust(bottom=0.15)
# plt.xlabel(u"th") #X轴标签
# plt.ylabel("srcc") #Y轴标签
# plt.title("th_srcc_feasure_curve") #标题
# plt.savefig('/home/w509/1workspace/lee/360_fix_sort/gtprove/curve_th_srcc/deepgaze2e_curve_th_srcc.jpg')
# plt.show()


#源代码

import numpy as  np
import h5py
import scipy.stats as stats
from util.rmse import rmse
import warnings

warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

srcc_list = []
plcc_list = []


train_dataset1 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/motivation/fixation/rank/val_multi_classi_0_0-0_75.h5', 'r')
train_dataset2 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/val_multi_classi_0_2-0_75.h5', 'r')

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

    eval_srcc += test_srcc
    eval_plcc += test_plcc
    eval_rmse += test_rmse
    eval_recall+=r
    eval_pre+=p
    eval_f1+=f1

print('=Srcc: {:.6f},Plcc: {:.6f},Rmse: {:.6f}'.format(eval_srcc / testlen, eval_plcc / testlen, eval_rmse / testlen))
print('=recall: {:.6f},precision: {:.6f},f1: {:.6f}'.format(eval_recall / testlen1, eval_pre / testlen1, eval_f1 / testlen1))



#












