import h5py
import numpy as  np
import os
import scipy.stats as stats
import cv2
method_name ="unisal"
txt_path = "/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/salicon_label_txt/select_val/vallocal/"
train_dataset = h5py.File('/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/val_multi_classi_0_5-0_75.h5', 'r')

train_dataset1 = h5py.File(
    '/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/{}/salicon_{}_val_multiclassi_new_0-5.h5'.format(method_name,method_name), 'r')
# train_dataset1 = h5py.File(
#     '/home/w509/1workspace/lee/360_fix_sort/gtprove/anothergt/true_true_true_gt/val_multi_classi_0_5-0_75.h5', 'r')
# train_dataset2 = h5py.File(
#     '/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/transalnet/salicon_transalnet_val_multiclassi_new_0-4.h5', 'r')
# train_dataset3 = h5py.File(
#     '/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/unisal/salicon_unisal_val_multiclassi_new_0-4.h5', 'r')
# train_dataset4 = h5py.File(
#     '/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/emlnet/salicon_emlnet_val_multiclassi_new_0-4.h5', 'r')
# train_dataset5 = h5py.File(
#     '/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/salgan/salicon_salgan_val_multiclassi_new_0-4.h5', 'r')
# train_dataset6 = h5py.File(
#     '/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/sam_resnet/salicon_sam_resnet_val_multiclassi_new_0-4.h5', 'r')
# train_dataset7 = h5py.File(
#     '/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/salicon/salicon_salicon_val_multiclassi_new_0-4.h5', 'r')
# train_dataset8 = h5py.File(
#     '/home/w509/1workspace/lee/360_fix_sort/otherRankModel/multiclassi_labels/mlnet/salicon_mlnet_val_multiclassi_new_0-4.h5', 'r')
train_labels = np.array(train_dataset['labels'][:])
train_labels = train_labels.tolist()
train_labels1 = np.array(train_dataset1['labels'][:])
train_labels1 = train_labels1.tolist()

# train_labels2 = np.array(train_dataset2['labels'][:])
# train_labels2 = train_labels2.tolist()
#
# train_labels3 = np.array(train_dataset3['labels'][:])
# train_labels3 = train_labels3.tolist()
# train_labels4 = np.array(train_dataset4['labels'][:])
# train_labels4 = train_labels4.tolist()
# train_labels5 = np.array(train_dataset5['labels'][:])
# train_labels5 = train_labels5.tolist()
# train_labels6 = np.array(train_dataset6['labels'][:])
# train_labels6 = train_labels6.tolist()
# train_labels7 = np.array(train_dataset7['labels'][:])
# train_labels7 = train_labels7.tolist()
# train_labels8 = np.array(train_dataset8['labels'][:])
# train_labels8 = train_labels8.tolist()



imgs_path ="/home/w509/1workspace/lee/360_fix_sort/runs/runs_datacate_unisal_resnet/image/val/"
# imgs_path ="/home/w509/1workspace/lee/360_fix_sort/duibitu/pic/"

gt_names = os.listdir(imgs_path)
# gt_name.sort(key=lambda x:int(x[11:-4]))
gt_names.sort(key=lambda x: int(x[:-4]))
# train_set = np.array(train_dataset['train'][:])


for gt_name in gt_names:
    print(gt_name)

    img = cv2.imread(imgs_path+gt_name)
    # cv2.imshow("imgs",img)
    # cv2.waitKey(0)
    i = int(gt_name[:-4])
    t = i
    txt_name=txt_path+"{:06d}".format(i)+".txt"
    i=(i-1)*5
    if t ==4073 or t==4164:
        pros = train_labels[i: i + 5]
        for ii, index in enumerate(pros):
            pros[ii] += 1
        pros.append(0)
        pros1 =train_labels1[i: i + 5]
        for ii, index in enumerate(pros1):
            pros1[ii] += 1
        pros1.append(0)
        # pros2 =train_labels2[i: i + 5]
        # for ii, index in enumerate(pros2):
        #     pros2[ii] += 1
        # pros2.append(0)
        # pros3 =train_labels3[i: i + 5]
        # for ii, index in enumerate(pros3):
        #     pros3[ii] += 1
        # pros3.append(0)
        # pros4 =train_labels4[i: i + 5]
        # for ii, index in enumerate(pros4):
        #     pros4[ii] += 1
        # pros4.append(0)
        # pros5 =train_labels5[i: i + 5]
        # for ii, index in enumerate(pros5):
        #     pros5[ii] += 1
        # pros5.append(0)
        # pros6 =train_labels6[i: i + 5]
        # for ii, index in enumerate(pros6):
        #     pros6[ii] += 1
        # pros6.append(0)
        # pros7 =train_labels7[i: i + 5]
        # for ii, index in enumerate(pros7):
        #     pros7[ii] += 1
        # pros7.append(0)
        # pros8 =train_labels8[i: i + 5]
        # for ii, index in enumerate(pros8):
        #     pros8[ii] += 1
        # pros8.append(0)
    else:
        pros = train_labels[i: i + 5]

        pros1 =train_labels1[i: i + 5]

        # pros2 =train_labels2[i: i + 5]
        #
        # pros3 =train_labels3[i: i + 5]
        #
        # pros4 =train_labels4[i: i + 5]
        #
        # pros5 =train_labels5[i: i + 5]
        #
        # pros6 =train_labels6[i: i + 5]
        #
        # pros7 =train_labels7[i: i + 5]
        #
        # pros8 =train_labels8[i: i + 5]


    f = open(txt_name, "r")
    index = 0
    while True:
        line = f.readline()
        if line:
            pass  # do something here
            line = line.strip()

            line = line.split(',')
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            # xcenter = line[4]
            # ycenter = line[5]
            # print(ycenter)

            x1 = int(float(x1))
            y1 = int(float(y1))
            x2 = int(float(x2))
            y2 = int(float(y2))
            xcenter = int((x1+x2)/2)
            ycenter = int((y1+y2)/2)

            print(pros1[index])
            text = pros1[index]

            text = str(text)

            label=pros[index]
            col = pros1[index]

            if label == col:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            img = cv2.putText(img, text, (xcenter, ycenter), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            index+=1
        else:
            cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/duibitu/draw/{}/{}'.format(method_name,gt_name), img)
            # cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/drawbox/1_pic.jpg', img_pic)
            break


    test_srcc1, _ = stats.spearmanr(pros, pros1)
    # test_srcc2, _ = stats.spearmanr(pros, pros2)
    # test_srcc3, _ = stats.spearmanr(pros, pros3)
    # test_srcc4, _ = stats.spearmanr(pros, pros5)
    # test_srcc5, _ = stats.spearmanr(pros, pros6)
    # test_srcc6, _ = stats.spearmanr(pros, pros6)
    # test_srcc7, _ = stats.spearmanr(pros, pros7)
    # test_srcc8, _ = stats.spearmanr(pros, pros8)

    print("salfbner:"+str(test_srcc1))
    # print("transalnet"+str(test_srcc2))
    # print("unisal"+str(test_srcc3))
    # print("emlnet"+str(test_srcc4))
    # print("sam_resnet"+str(test_srcc5))
    # print("salgan"+str(test_srcc6))
    # print("salicon"+str(test_srcc7))
    # print("mlnet"+str(test_srcc8))










