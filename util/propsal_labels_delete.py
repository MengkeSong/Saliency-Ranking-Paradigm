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
fakenum =h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/h5_blackdot_nums/fake_nums.h5','r')
fake_nums= np.array(fakenum['labels'][:])
fake_nums = fake_nums.tolist()

unisal_num = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/h5_blackdot_nums/multiclassi_labels/unisal_val_multiclassi_0-4.h5', 'r')
unisal_nums= np.array(unisal_num['labels'][:])
unisal_nums = unisal_nums.tolist()


unisal_labels= []
ids = 0
for i in range(0, len(unisal_nums), 5):
    pros = unisal_nums[i: i + 5]
    pros_2classifer = fake_nums[ids:ids+2]
    ids+=2
    alljpg_pro = pros_2classifer[0]
    annther_pro = pros_2classifer[1]

    pros_another = pros[:-annther_pro]
    if len(pros_another)>0:
        pros_labels = paixu(pros_another)
        if alljpg_pro ==0:
            result_labels = [5]
            if annther_pro ==0:

                labels = []
                pros_labels.extend(labels)
                result_labels.extend(pros_labels)
            if annther_pro == 1:
                labels = [0]
                pros_labels.extend(labels)
                result_labels.extend(pros_labels)

            if annther_pro ==2:
                labels = [1,0]
                pros_labels.extend(labels)
                result_labels.extend(pros_labels)

            if annther_pro ==3:
                labels= [2,1,0]
                pros_labels.extend(labels)
                result_labels.extend(pros_labels)

            if annther_pro ==4:
                labels=[3,2,1,0]
                pros_labels.extend(labels)
                result_labels.extend(pros_labels)

            if annther_pro ==5:
                labels = [4,3,2,1,0]
                pros_labels.extend(labels)
                result_labels.extend(pros_labels)

        elif alljpg_pro ==1:
            result_labels = []
            if annther_pro == 0:
                labels = []
                result_labels.append(annther_pro)
                pros_labels.extend(labels)
                result_labels.extend(pros_labels)


            if annther_pro == 1:
                labels = [0]
                result_labels.append(annther_pro)

                pros_labels.extend(labels)
                result_labels.extend(pros_labels)

            if annther_pro == 2:
                labels = [1, 0]
                result_labels.append(annther_pro)

                pros_labels.extend(labels)
                result_labels.extend(pros_labels)

            if annther_pro == 3:
                labels = [2, 1, 0]
                result_labels.append(annther_pro)

                pros_labels.extend(labels)
                result_labels.extend(pros_labels)

            if annther_pro == 4:
                labels = [3, 2, 1, 0]
                result_labels.append(annther_pro)

                pros_labels.extend(labels)
                result_labels.extend(pros_labels)

            if annther_pro == 5:
                labels = [4, 3, 2, 1, 0]
                result_labels.append(annther_pro)

                pros_labels.extend(labels)
                result_labels.extend(pros_labels)
        unisal_labels.extend(result_labels)
        # print(len(result_labels))
        # print(ids)
    else:
        pros_labels = []
        result_labels = [5,4,3,2,1,0]
        # if alljpg_pro == 0:
        #     result_labels = [5]
        #     if annther_pro == 0:
        #         labels = []
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #     if annther_pro == 1:
        #         labels = [0]
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        #     if annther_pro == 2:
        #         labels = [1, 0]
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        #     if annther_pro == 3:
        #         labels = [2, 1, 0]
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        #     if annther_pro == 4:
        #         labels = [3, 2, 1, 0]
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        #     if annther_pro == 5:
        #         labels = [4, 3, 2, 1, 0]
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        # elif alljpg_pro == 1:
        #     if ids==2336:
        #         print(1)
        #     result_labels = []
        #     if annther_pro == 0:
        #         labels = []
        #         result_labels.append(annther_pro)
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        #     if annther_pro == 1:
        #         labels = [0]
        #         result_labels.append(annther_pro)
        #
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        #     if annther_pro == 2:
        #         labels = [1, 0]
        #         result_labels.append(annther_pro)
        #
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        #     if annther_pro == 3:
        #         labels = [2, 1, 0]
        #         result_labels.append(annther_pro)
        #
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        #     if annther_pro == 4:
        #         labels = [3, 2, 1, 0]
        #         result_labels.append(annther_pro)
        #
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        #
        #     if annther_pro == 5:
        #         labels = [4, 3, 2, 1, 0]
        #         result_labels.append(annther_pro)
        #
        #         pros_labels.extend(labels)
        #         result_labels.extend(pros_labels)
        unisal_labels.extend(result_labels)
        # print(len(result_labels))
        #
        # print(ids)

unisal_labels= np.array(unisal_labels)
print(unisal_labels.shape)

f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/h5_blackdot_nums/multiclassi_labels/unisal_val_multiclassi_0-5.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=unisal_labels)
f.close()

