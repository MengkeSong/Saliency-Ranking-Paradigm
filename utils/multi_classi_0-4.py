# import numpy as  np
# import h5py
# for t1 in range(0,2):
#     for t2 in range(0,10):
#
#         train_dataset = h5py.File('/home/w509/1workspace/lee/360_fix_sort/ranking_model/h5_blackdot_nums/deepgaze2e/salicon_deepgaze2e_val_multiclassi_new_{}-{}.h5'.format(t1,t2), 'r')
#
#         # train_set = np.array(train_dataset['train'][:])
#         train_labels=np.array(train_dataset['labels'][:])
#         train_labels=train_labels.tolist()
#         train_label=[]
#         for i in range(0, len(train_labels), 5):
#             pros = train_labels[i: i + 5]
#             aa=pros
#             bb = [(aa[i - 1], i) for i in range(1, len(aa) + 1)]
#             cc = sorted(bb)
#             dd = [(cc[i - 1][0], i, cc[i - 1][1]) for i in range(1, 6)]
#             ee = sorted(dd, key=lambda x: x[2])
#             ff = [x[1] for x in ee]
#             ff_copy=[(x-1) for x in ff]
#
#
#             ff_copy_copy = ff_copy.copy()
#             # for z in range(3):
#             #     if ff_copy[z] == 1:
#             #         ff_copy_copy[z] += 4
#             #     if ff_copy[z] == 2:
#             #         ff_copy_copy[z] += 7
#
#             # for z in range(3):
#             #     if pros[z] == 0:
#             #         ff_copy_copy[z] = 0
#
#             train_label.extend(ff_copy_copy)
#
#
#         train_label= np.array(train_label)
#         print(train_label.shape)
#         f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/ranking_model/h5_blackdot_nums/multiclassi_labels/deepgaze2e/salicon_deepgaze2e_val_multiclassi_new_{}-{}.h5'.format(t1,t2), 'w')
#         # f.create_dataset('train', data=train_set)
#         f.create_dataset('labels', data=train_label)
#         f.close()


import numpy as  np
import h5py

train_dataset = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/val_nums_1_0-0_75.h5', 'r')

# train_set = np.array(train_dataset['train'][:])
train_labels=np.array(train_dataset['labels'][:])
train_labels=train_labels.tolist()
train_label=[]
for i in range(0, len(train_labels), 5):
    pros = train_labels[i: i + 5]
    aa=pros
    bb = [(aa[i - 1], i) for i in range(1, len(aa) + 1)]
    cc = sorted(bb)
    dd = [(cc[i - 1][0], i, cc[i - 1][1]) for i in range(1, 6)]
    ee = sorted(dd, key=lambda x: x[2])
    ff = [x[1] for x in ee]
    ff_copy=[(x-1) for x in ff]


    ff_copy_copy = ff_copy.copy()
    # for z in range(3):
    #     if ff_copy[z] == 1:
    #         ff_copy_copy[z] += 4
    #     if ff_copy[z] == 2:
    #         ff_copy_copy[z] += 7

    # for z in range(3):
    #     if pros[z] == 0:
    #         ff_copy_copy[z] = 0

    train_label.extend(ff_copy_copy)


train_label= np.array(train_label)
print(train_label.shape)
f = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/val_multi_classi_1_0-0_75.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=train_label)
f.close()

