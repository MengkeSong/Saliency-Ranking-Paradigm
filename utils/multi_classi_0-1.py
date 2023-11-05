import numpy as  np
import h5py

train_dataset = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/h5_blackdot_nums/multiclassi_labels/train_multiclassi_0-4.h5', 'r')

# train_set = np.array(train_dataset['train'][:])
train_labels=np.array(train_dataset['labels'][:])
train_labels=train_labels.tolist()
train_label=[]
for i in range(0, len(train_labels), 5):
    pros = train_labels[i: i + 5]
    ff_copy=[]
    for z in pros:
        if z < 3:
            label = 0
            ff_copy.append(label)
        else:
            label = 1
            ff_copy.append(label)
    ff_copy.append(1)





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
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut-omron-new/h5_blackdot_nums/multiclassi_labels/train_multiclassi_0-1_change.h5', 'w')
# f.create_dataset('train', data=train_set)
f.create_dataset('labels', data=train_label)
f.close()
