import numpy as  np
import h5py
import scipy.stats as stats
from util.rmse import rmse
import warnings
import xlwt

warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
gt_path="/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/val_multi_classi_0_2-0_75.h5"
d_name="salicon"
srcc_list = []
f1_list = []
acc_first_list = []
acc_sec_list = []
acc_thi_list = []
result = xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet = result.add_sheet('jieguo',cell_overwrite_ok=True)
col = ('f1','srcc','acc-1','acc-2','acc-3')
for i in range(0,5):
		sheet.write(0,i,col[i])


for t1 in range(0,2):
    if t1 ==0:
        for t2 in range(0,10):


            train_dataset1 = h5py.File('/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/{}/salicon_{}_val_multiclassi_new_{}-{}.h5'.format(d_name,d_name,t1,t2), 'r')
            train_dataset2 = h5py.File(gt_path, 'r')

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
            nums_first = 0
            nums_second = 0
            nums_third = 0
            for i in range(0, len(train_labels1), 5):
                pros1 = train_labels1[i: i + 5]
                pros2 = train_labels2[i: i + 5]

                test_srcc, _ = stats.spearmanr(pros1, pros2)
                test_plcc, _ = stats.pearsonr(pros1, pros2)
                test_rmse = rmse(pros1, pros2)

                f1 = f1_score(pros2, pros1, average='macro')
                p = precision_score(pros2, pros1, average='macro')
                r = recall_score(pros2, pros1, average='macro')

                index_gt1 = pros2.index(max(pros2))
                if pros2[index_gt1] == pros1[index_gt1]:
                    nums_first += 1
                del pros2[index_gt1]
                del pros1[index_gt1]

                index_gt2 = pros2.index(max(pros2))
                if pros2[index_gt2] == pros1[index_gt2]:
                    nums_second += 1
                del pros2[index_gt2]
                del pros1[index_gt2]

                index_gt3 = pros2.index(max(pros2))
                if pros2[index_gt3] == pros1[index_gt3]:
                    nums_third += 1
                del pros2[index_gt3]
                del pros1[index_gt3]


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
            srcc_list.append(round(eval_srcc / testlen,3))
            f1_list.append(round(eval_f1 / testlen,3))
            acc_first = round(nums_first / testlen,3)
            acc_second = round(nums_second / testlen,3)
            acc_third = round(nums_third / testlen,3)
            acc_first_list.append(acc_first)
            acc_sec_list.append(acc_second)
            acc_thi_list.append(acc_third)
            print('=Srcc: {:.6f},Plcc: {:.6f},Rmse: {:.6f}'.format(eval_srcc / testlen, eval_plcc / testlen, eval_rmse / testlen))
            print('=recall: {:.6f},precision: {:.6f},f1: {:.6f}'.format(eval_recall / testlen1, eval_pre / testlen1, eval_f1 / testlen1))
            print('=acc_first: {:.6f},acc_sec: {:.6f},acc_third: {:.6f}'.format(acc_first, acc_second,
                                                                        acc_third))
    else:
        for t2 in range(0, 1):

            train_dataset1 = h5py.File(
                '/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/{}/salicon_{}_val_multiclassi_new_{}-{}.h5'.format(
                    d_name,d_name,t1, t2), 'r')
            train_dataset2 = h5py.File(
                gt_path,
                'r')

            train_labels1 = np.array(train_dataset1['labels'][:])
            train_labels1 = train_labels1.tolist()

            train_labels2 = np.array(train_dataset2['labels'][:])
            train_labels2 = train_labels2.tolist()

            testlen = (len(train_labels1) / 5)
            testlen1 = (len(train_labels1) / 5)

            eval_srcc = 0
            eval_plcc = 0
            eval_rmse = 0
            eval_recall = 0
            eval_pre = 0
            eval_f1 = 0
            nums_first = 0
            nums_second = 0
            nums_third = 0
            for i in range(0, len(train_labels1), 5):
                pros1 = train_labels1[i: i + 5]
                pros2 = train_labels2[i: i + 5]

                test_srcc, _ = stats.spearmanr(pros1, pros2)
                test_plcc, _ = stats.pearsonr(pros1, pros2)
                test_rmse = rmse(pros1, pros2)

                f1 = f1_score(pros2, pros1, average='macro')
                p = precision_score(pros2, pros1, average='macro')
                r = recall_score(pros2, pros1, average='macro')

                index_gt1 = pros2.index(max(pros2))
                if pros2[index_gt1] == pros1[index_gt1]:
                    nums_first += 1
                del pros2[index_gt1]
                del pros1[index_gt1]

                index_gt2 = pros2.index(max(pros2))
                if pros2[index_gt2] == pros1[index_gt2]:
                    nums_second += 1
                del pros2[index_gt2]
                del pros1[index_gt2]

                index_gt3 = pros2.index(max(pros2))
                if pros2[index_gt3] == pros1[index_gt3]:
                    nums_third += 1
                del pros2[index_gt3]
                del pros1[index_gt3]

                if np.isnan(test_srcc) or np.isnan(test_plcc):
                    test_srcc = 0
                    test_plcc = 0
                    testlen -= 1

                eval_srcc += test_srcc
                eval_plcc += test_plcc
                eval_rmse += test_rmse
                eval_recall += r
                eval_pre += p
                eval_f1 += f1
            # f1_list.append(eval_f1 / testlen)
            # srcc_list.append(eval_srcc / testlen)
            # acc_first = nums_first / testlen
            # acc_second = nums_second / testlen
            # acc_third = nums_third / testlen
            srcc_list.append(round(eval_srcc / testlen,3))
            f1_list.append(round(eval_f1 / testlen,3))
            acc_first = round(nums_first / testlen,3)
            acc_second = round(nums_second / testlen,3)
            acc_third = round(nums_third / testlen,3)
            acc_first_list.append(acc_first)
            acc_sec_list.append(acc_second)
            acc_thi_list.append(acc_third)
            print('=Srcc: {:.6f},Plcc: {:.6f},Rmse: {:.6f}'.format(eval_srcc / testlen, eval_plcc / testlen,
                                                                   eval_rmse / testlen))
            print('=recall: {:.6f},precision: {:.6f},f1: {:.6f}'.format(eval_recall / testlen1, eval_pre / testlen1,
                                                                        eval_f1 / testlen1))
            print('=acc_first: {:.6f},acc_sec: {:.6f},acc_third: {:.6f}'.format(acc_first, acc_second,
                                                                        acc_third))
print("f1")
print(f1_list)
print("srcc")
print(srcc_list)
print("acc_first")
print(acc_first_list)
print("acc_sec")
print(acc_sec_list)
print("acc_third")
print(acc_thi_list)
import matplotlib.pyplot as plt
# f1_list=[0.618, 0.657, 0.674, 0.682, 0.685, 0.68, 0.674, 0.658, 0.64, 0.624, 0.604]
# srcc_list=[0.819, 0.847, 0.861, 0.868, 0.868, 0.864, 0.859, 0.847, 0.833, 0.817, 0.798]
# acc_first_list=[0.783, 0.813, 0.828, 0.839, 0.847, 0.851, 0.851, 0.846, 0.839, 0.833, 0.819]
# acc_sec_list=[0.587, 0.627, 0.644, 0.663, 0.675, 0.681, 0.681, 0.674, 0.663, 0.648, 0.628]
# acc_thi_list=[0.501, 0.546, 0.567, 0.585, 0.588, 0.586, 0.584, 0.566, 0.546, 0.53, 0.508]
datalist = []
for i in range(0,11):
    datalist = []
    f1_ = f1_list[i]
    datalist.append(f1_)
    srcc_ = srcc_list[i]
    datalist.append(srcc_)
    acc1_ = acc_first_list[i]
    datalist.append(acc1_)
    acc2_ = acc_sec_list[i]
    datalist.append(acc2_)
    acc3_ = acc_thi_list[i]
    datalist.append(acc3_)
    for j in range(0,5):
        sheet.write(i+1,j,datalist[j])
savepath = '/home/w509/1workspace/lee/360_fix_sort/excel/excel_0-2/excel_{}.xls'.format(d_name)
result.save(savepath)


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

# y = [0.2786,0.09526,0.03631,0.01232,0.00291]

# y1=[0.452,0.1272,0.03737902,0.01120912,0.00281906]
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

