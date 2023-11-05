import numpy as np
import h5py

def paixu(train_labels):
    train_label = []
    lens = len(train_labels)
    for i in range(0, len(train_labels), lens):
        pros = train_labels[i: i + lens]
        aa=pros
        bb = [(aa[i - 1], i) for i in range(1, len(aa) + 1)]
        cc = sorted(bb)
        dd = [(cc[i - 1][0], i, cc[i - 1][1]) for i in range(1, lens+1)]
        ee = sorted(dd, key=lambda x: x[2])
        ff = [x[1] for x in ee]
        ff_copy=[(x-1) for x in ff]


        ff_copy_copy = ff_copy.copy()

        if lens == 1:
            for t in range(len(ff_copy)):
                ff_copy_copy[t]+=5
        if lens == 2:
            for t in range(len(ff_copy)):
                ff_copy_copy[t]+=4
        if lens == 3:
            for t in range(len(ff_copy)):
                ff_copy_copy[t]+=3
        if lens == 4:
            for t in range(len(ff_copy)):
                ff_copy_copy[t]+=2
        if lens == 5:
            for t in range(len(ff_copy)):
                ff_copy_copy[t]+=1

    return ff_copy_copy








trainlabel =h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/h5_blackdot_nums/val_nums.h5','r')
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
select_fea1 = []
select_fea2 = []
select_x = []
select_y = []
select_la =[]
fake_nums = []

# fake_num = 0
ids = 0
for i in range(0, len(labels_set), 5):
    pros = labels_set[i: i + 5]
    pros_2classifer = classifier2_nums[ids:ids+6]
    pros_unisal = unisal_nums[i:i+5]
    feature1 = train_set_feature1[ids:ids+6]
    feature2 = train_set_feature2[ids:ids+6]
    xcenter = train_set_xcenter[ids:ids+6]
    ycenter = train_set_ycenter[ids:ids+6]
    select_train_feature1 = []
    select_train_feature2 = []
    select_xcenter = []
    select_ycenter = []
    select_labels = []

#寻找propsal中有fixation点的特征
    for t in range(0,5):
        if pros_2classifer[t] ==1 and pros_unisal[t]>100:
            select_train_feature1.append(feature1[t])
            select_train_feature2.append(feature2[t])
            select_xcenter.append(xcenter[t])
            select_ycenter.append(ycenter[t])
            select_labels.append(pros[t])
    if len(select_labels)>0:
        fake_nums.append(1)
        select_labels = paixu(select_labels)
        labels_len = len(select_labels)
        if labels_len==1:
            select_train_feature1.append(feature1[5])
            select_train_feature2.append(feature2[5])
            select_xcenter.append(xcenter[5])
            select_ycenter.append(ycenter[5])
            select_labels.append(4)
            fake_nums.append(4)
            for tt in [3,2,1,0]:
                fakefeature1 = np.full((1,192,7,7),tt,dtype=float)
                fakefeature1 = fakefeature1.tolist()
                fakefeature2 = np.full((1,192,7,7),tt,dtype=float)
                fakefeature2= fakefeature2.tolist()
                fakexcenter = xcenter[5]
                fakeycenter = ycenter[5]
                fakelabel = tt

                select_train_feature1.append(fakefeature1)
                select_train_feature2.append(fakefeature2)
                select_xcenter.append(fakexcenter)
                select_ycenter.append(fakeycenter)
                select_labels.append(fakelabel)


        if labels_len==2:
            select_train_feature1.append(feature1[5])
            select_train_feature2.append(feature2[5])
            select_xcenter.append(xcenter[5])
            select_ycenter.append(ycenter[5])
            select_labels.append(3)
            fake_nums.append(3)

            for tt in [2, 1, 0]:
                fakefeature1 = np.full((1, 192, 7, 7), tt,dtype=float)
                fakefeature1 = fakefeature1.tolist()
                fakefeature2 = np.full((1, 192, 7, 7), tt,dtype=float)
                fakefeature2 = fakefeature2.tolist()
                fakexcenter = xcenter[5]
                fakeycenter = ycenter[5]
                fakelabel = tt

                select_train_feature1.append(fakefeature1)
                select_train_feature2.append(fakefeature2)
                select_xcenter.append(fakexcenter)
                select_ycenter.append(fakeycenter)
                select_labels.append(fakelabel)

        if labels_len==3:
            select_train_feature1.append(feature1[5])
            select_train_feature2.append(feature2[5])
            select_xcenter.append(xcenter[5])
            select_ycenter.append(ycenter[5])
            select_labels.append(2)
            fake_nums.append(2)
            for tt in [1, 0]:
                fakefeature1 = np.full((1, 192, 7, 7), tt,dtype=float)
                fakefeature1 = fakefeature1.tolist()
                fakefeature2 = np.full((1, 192, 7, 7), tt,dtype=float)
                fakefeature2 = fakefeature2.tolist()
                fakexcenter = xcenter[5]
                fakeycenter = ycenter[5]
                fakelabel = tt

                select_train_feature1.append(fakefeature1)
                select_train_feature2.append(fakefeature2)
                select_xcenter.append(fakexcenter)
                select_ycenter.append(fakeycenter)
                select_labels.append(fakelabel)

        if labels_len==4:
            select_train_feature1.append(feature1[5])
            select_train_feature2.append(feature2[5])
            select_xcenter.append(xcenter[5])
            select_ycenter.append(ycenter[5])
            select_labels.append(1)
            fake_nums.append(1)
            for tt in [0]:
                fakefeature1 = np.full((1, 192, 7, 7), tt,dtype=float)
                fakefeature1 = fakefeature1.tolist()
                fakefeature2 = np.full((1, 192, 7, 7), tt,dtype=float)
                fakefeature2 = fakefeature2.tolist()
                fakexcenter = xcenter[5]
                fakeycenter = ycenter[5]
                fakelabel = tt

                select_train_feature1.append(fakefeature1)
                select_train_feature2.append(fakefeature2)
                select_xcenter.append(fakexcenter)
                select_ycenter.append(fakeycenter)
                select_labels.append(fakelabel)

        if labels_len==5:
            select_train_feature1.append(feature1[5])
            select_train_feature2.append(feature2[5])
            select_xcenter.append(xcenter[5])
            select_ycenter.append(ycenter[5])
            select_labels.append(0)
            fake_nums.append(0)

    else:
        fake_nums.append(0)
        select_train_feature1.append(feature1[5])
        select_train_feature2.append(feature2[5])
        select_xcenter.append(xcenter[5])
        select_ycenter.append(ycenter[5])
        select_labels.append(5)
        fake_nums.append(5)
        for tt in [4,3, 2, 1, 0]:
            fakefeature1 = np.full((1, 192, 7, 7), tt,dtype=float)
            fakefeature1 = fakefeature1.tolist()
            fakefeature2 = np.full((1, 192, 7, 7), tt,dtype=float)
            fakefeature2 = fakefeature2.tolist()
            fakexcenter = xcenter[5]
            fakeycenter = ycenter[5]
            fakelabel = tt

            select_train_feature1.append(fakefeature1)
            select_train_feature2.append(fakefeature2)
            select_xcenter.append(fakexcenter)
            select_ycenter.append(fakeycenter)
            select_labels.append(fakelabel)



    #将图片自身加入
    select_fea1.extend(select_train_feature1)
    select_fea2.extend(select_train_feature2)
    select_x.extend(select_xcenter)
    select_y.extend(select_ycenter)
    select_la.extend(select_labels)
    ids+=6


select_fea1=np.array(select_fea1)
select_fea2=np.array(select_fea2)
select_x = np.array(select_x)
select_y = np.array(select_y)
select_la= np.array(select_la)
fake_nums = np.array(fake_nums)
print(select_fea2.shape)
print(select_x.shape)
print(select_la.shape)

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/h5_blackdot_nums/multiclassi_labels/val_multiclassi_0-5—100.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=select_la)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/h5_blackdot_nums/fake_nums.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=fake_nums)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/val_unisal_roi_local.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('train', data=select_fea1)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/val_unisal_roi_global.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('train', data=select_fea2)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/xycenter/xcenter_val_adjpg.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('xcenter', data=select_x)
f.close()

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/xycenter/ycenter_val_adjpg.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('ycenter', data=select_y)
f.close()



