import numpy as np
import h5py

trainlabel =h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/h5_blackdot_nums/multiclassi_labels/val_multiclassi_0-4.h5','r')
trainfeature1 =h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/unisal_clone/feature_dut_2classi/val_unisal_roi_local.h5','r')
trainfeature2 =h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/unisal_clone/feature_dut_2classi/val_unisal_roi_global.h5','r')
train_xcenter = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/xycenter/xcenter_val_adjpg.h5', 'r')
train_ycenter = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/xycenter/ycenter_val_adjpg.h5', 'r')
unisal_num = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/h5_blackdot_nums/unisal_valnums_new_0-5.h5', 'r')
classifier2_num = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/pred_save/pred.h5', 'r')

unisal_nums= np.array(unisal_num['labels'][:])
unisal_nums = unisal_nums.tolist()

classifier2_nums= np.array(classifier2_num['labels'][:])
classifier2_nums = classifier2_nums.tolist()


train_set_feature1 = np.array(trainfeature1['train'][:])
train_set_feature1 = train_set_feature1.tolist()
train_set_feature2 = np.array(trainfeature2['train'][:])
train_set_feature2 = train_set_feature2.tolist()
# train_set2 = np.array(train_dataset2['train'][:])
train_set_xcenter = np.array(train_xcenter['xcenter'][:])
train_set_xcenter = train_set_xcenter.tolist()
train_set_ycenter = np.array(train_ycenter['ycenter'][:])
train_set_ycenter = train_set_ycenter.tolist()
# train_set5 = np.array(train_dataset5['labels'][:])
labels_set = np.array(trainlabel['labels'][:])
labels_set = labels_set.tolist()


select_train_feature1 = []
select_train_feature2 = []
select_xcenter = []
select_ycenter = []
select_labels = []
select_uni_nums = []
ids = 0
for i in range(0, len(labels_set), 5):
    pros = labels_set[i: i + 5]
    feature1 = train_set_feature1[ids:ids+6]
    feature2 = train_set_feature2[ids:ids+6]
    xcenter = train_set_xcenter[ids:ids+6]
    ycenter = train_set_ycenter[ids:ids+6]
    uni_nums = unisal_nums[i:i+5]
    cla2_nums = classifier2_nums[ids:ids+6]

    zz = 0

#寻找propsal中的排名前2的特征
    for t in range(len(uni_nums)):
        if uni_nums[t] >0 and cla2_nums[t] ==1 and zz<=1:
            zz+=1
            select_train_feature1.append(feature1[t])
            select_train_feature2.append(feature2[t])
            select_xcenter.append(xcenter[t])
            select_ycenter.append(ycenter[t])
            select_labels.append(pros[t])
            select_uni_nums.append(uni_nums[t])

    if zz==0:
        # feature11 = feature1.copy()
        # feature22 = feature2.copy()
        # xcenter11 = xcenter.copy()
        # ycenter11 = ycenter.copy()
        # pros11 = pros.copy()
        # uni_nums11 = uni_nums.copy()
        # t = uni_nums11.index(max(uni_nums11))
        # select_train_feature1.append(feature11[t])
        # select_train_feature2.append(feature22[t])
        # select_xcenter.append(xcenter11[t])
        # select_ycenter.append(ycenter11[t])
        # select_labels.append(pros11[t])
        # select_uni_nums.append(uni_nums11[t])
        # del uni_nums11[t]
        # del feature11[t]
        # del feature22[t]
        # del xcenter11[t]
        # del ycenter11[t]
        # del pros11[t]
        # t = uni_nums11.index(max(uni_nums11))
        # select_train_feature1.append(feature11[t])
        # select_train_feature2.append(feature22[t])
        # select_xcenter.append(xcenter11[t])
        # select_ycenter.append(ycenter11[t])
        # select_labels.append(pros11[t])
        # select_uni_nums.append(uni_nums11[t])
        t =0
        select_train_feature1.append(feature1[t])
        select_train_feature2.append(feature2[t])
        select_xcenter.append(xcenter[t])
        select_ycenter.append(ycenter[t])
        select_labels.append(pros[t])
        select_uni_nums.append(uni_nums[t])

        t = 1
        select_train_feature1.append(feature1[t])
        select_train_feature2.append(feature2[t])
        select_xcenter.append(xcenter[t])
        select_ycenter.append(ycenter[t])
        select_labels.append(pros[t])
        select_uni_nums.append(uni_nums[t])

    if zz==1:
        if t<=4 and t>0:
            t-=1
            select_train_feature1.append(feature1[t])
            select_train_feature2.append(feature2[t])
            select_xcenter.append(xcenter[t])
            select_ycenter.append(ycenter[t])
            select_labels.append(pros[t])
            select_uni_nums.append(uni_nums[t])

        else:
            t+=1
            select_train_feature1.append(feature1[t])
            select_train_feature2.append(feature2[t])
            select_xcenter.append(xcenter[t])
            select_ycenter.append(ycenter[t])
            select_labels.append(pros[t])
            select_uni_nums.append(uni_nums[t])





        #将图片自身加入
    select_train_feature1.append(feature1[5])
    select_train_feature2.append(feature2[5])
    select_xcenter.append(xcenter[5])
    select_ycenter.append(ycenter[5])
    select_labels.append(5)
    select_uni_nums.append(1000000000)

    ids+=6


select_train_feature1=np.array(select_train_feature1)
select_train_feature2=np.array(select_train_feature2)
select_xcenter = np.array(select_xcenter)
select_ycenter = np.array(select_ycenter)
select_labels = np.array(select_labels)
select_uni_nums = np.array(select_uni_nums)

print(select_train_feature1.shape)
print(select_xcenter.shape)
print(select_labels.shape)

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select_demo/h5_blackdot_nums/val_nums.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=select_labels)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select_demo/h5_blackdot_nums/val_unisal_nums.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=select_uni_nums)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select_demo/val_unisal_roi_local.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('train', data=select_train_feature1)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select_demo/val_unisal_roi_global.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('train', data=select_train_feature2)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select_demo/xycenter/xcenter_val_adjpg.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('xcenter', data=select_xcenter)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select_demo/xycenter/ycenter_val_adjpg.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('ycenter', data=select_ycenter)
f.close()



