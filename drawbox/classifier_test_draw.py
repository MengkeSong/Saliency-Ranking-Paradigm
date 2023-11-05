import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as datas
from torchsummary import summary
import scipy.stats as stats
from utils.rmse import rmse
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
# from  net import GCN_Net,PyGCN
import model
import cv2
from model.net4multiclassi_unisal import GCN_Net
from torch.optim import *

print(os.listdir("./"))
if __name__ == "__main__":

    EPOCHS = 2000
    BATCH_SIZE = 5
    learning_rate = 0.00001
    txt_path = '/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/boxtxt/vallocal/'
    img_path = '/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/2dfixdataset/images/val/'
    imgs_path = os.listdir(img_path)
    imgs_path.sort(key=lambda x: int(x[:-4]))
    txts_path = os.listdir(txt_path)
    txts_path.sort(key=lambda x: int(x[:-8]))
    import torch.utils.data as data
    import torch
    import h5py


    class DataFromH5File(data.Dataset):
        def __init__(self, filepath):
            h5File = h5py.File(filepath, 'r')
            self.data1 = h5File['train1']
            self.data2 = h5File['train2']
            self.xcenter = h5File['xcenter']
            self.ycenter = h5File['ycenter']

            self.labels = h5File['labels']

        def __getitem__(self, idx):
            label = torch.tensor(self.labels[idx]).float()
            data1 = torch.tensor(self.data1[idx]).float()
            data2 = torch.tensor(self.data2[idx]).float()
            xcenter = torch.tensor(self.xcenter[idx]).float()
            ycenter = torch.tensor(self.ycenter[idx]).float()


            return data1,data2,xcenter,ycenter, label

        def __len__(self):
            assert self.data1.shape[0] == self.labels.shape[0], "Wrong data length"
            return self.data1.shape[0]


    class DatatestFromH5File(data.Dataset):
        def __init__(self, filepath):
            h5File = h5py.File(filepath, 'r')
            self.data1 = h5File['train1']
            self.data2 = h5File['train2']
            self.xcenter = h5File['xcenter']
            self.ycenter = h5File['ycenter']
            self.labels = h5File['labels']

        def __getitem__(self, idx):
            label = torch.tensor(self.labels[idx]).float()
            data1 = torch.tensor(self.data1[idx]).float()
            data2 = torch.tensor(self.data2[idx]).float()
            xcenter = torch.tensor(self.xcenter[idx]).float()
            ycenter = torch.tensor(self.ycenter[idx]).float()


            return data1,data2,xcenter,ycenter, label

        def __len__(self):
            assert self.data1.shape[0] == self.labels.shape[0], "Wrong data length"
            return self.data1.shape[0]


    def isnull(list):
        gt_id = 0
        for gt in list:
            if gt > 0:
                gt_id = 1
        return gt_id

    trainset = DataFromH5File(
        "/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/train_val/train_local_global_unisal_multi_2classi_fixs.h5")
    train_loader = data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    testset = DatatestFromH5File(
        "/home/w509/1workspace/lee/2dfix_classi/feature_salcon/train_val_unisal/val_local_global_unisal_multi_classi_fixs_true.h5")
    test_loader = data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    labels_dataset_unisal = h5py.File('/home/w509/1workspace/lee/2dfix_classi/feature_salcon/h5_blackdot_nums/multiclassi_labels/salicon_unisal_val_multiclassi_new_1-5.h5', 'r')
    labels_dataset_salicon = h5py.File('/home/w509/1workspace/lee/2dfix_classi/feature_salcon/h5_blackdot_nums/multiclassi_labels/salicon_val_multiclassi_new_1_5.h5', 'r')
    labels_dataset_eml = h5py.File('/home/w509/1workspace/lee/ranking_model/h5_blackdot_nums/multiclassi_labels/salicon_emlnet_val_multiclassi_new_1_5.h5', 'r')
    labels_dataset_salgan = h5py.File('/home/w509/1workspace/lee/ranking_model/h5_blackdot_nums/multiclassi_labels/salicon_salgan_val_multiclassi_new_1_5.h5', 'r')

    labels_set_unisal = np.array(labels_dataset_unisal['labels'][:])
    labels_set_eml = np.array(labels_dataset_eml['labels'][:])
    labels_set_salicon = np.array(labels_dataset_salicon['labels'][:])
    labels_set_salgan = np.array(labels_dataset_salgan['labels'][:])

    model = GCN_Net()

    for param in model.parameters():
        param.requires_grad = True


    total_params = sum(p.numel() for p in model.parameters())
    print('Number of Parameters: {}'.format(total_params))
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Training Parameters: {}'.format(total_trainable_params))

    """
    Data to CUDA device
    """

    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()
    model.load_state_dict(torch.load('/home/w509/1workspace/lee/360_fix_sort/checkpoint/model_salicon_unisal_ranking_true.pt'))
    # model.load_state_dict(torch.load('/home/w509/1workspace/lee/2dfix_classi/checkpoint/model_multi_classi_global_local_sal_flo.pt'))




    print(40 * "-")
    # %%
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005, nesterov=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15,20,25], gamma=0.8)


    # optimizer = torch.optim.Adam(model.parameters())
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    torch.cuda.empty_cache()

    valid_max_acc = 0
    history = []

    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0

    overall_start = timer()
    nums = 0
    eval_bestacc = 0
    best_epoch = 0
    eval_bestsrcc = 0
    bestsrcc = 0

    best_acc_first = 0

    with torch.no_grad():
        model.eval()  # 模型评估
        eval_loss = 0
        eval_acc = 0
        eval_srcc = 0
        eval_plcc = 0
        eval_rmse = 0
        testlen_first = 0
        testlen_second = 0
        testlen_third = 0
        testlen = (len(testset) / 5)

        nums_first = 0
        nums_second = 0
        nums_third = 0
        save = []

        # path = r'//home/w509/1workspace/lee/360_fix_sort/boxtxt/sitzmann/test/'
        # imgs_path = os.listdir(path)
        # imgs_path.sort(key=lambda x: int(x[:-4]))
        for ii, (data1,data2,xcenter,ycenter, labels) in enumerate(test_loader):  # 测试模型
            if CUDA:
                data1, data2, xcenter, ycenter, labels = data1.cuda(), data2.cuda(), xcenter.cuda(), ycenter.cuda(), labels.cuda()

            optimizer.zero_grad()
            torch.cuda.empty_cache()





            out = model(data1,data2,xcenter,ycenter)

            loss = criterion(out, labels.long())



            eval_loss += loss.item() * labels.size(0)

            cost = torch.exp(-out)

            row_ind, col_ind = linear_sum_assignment(cost.cpu())
            # if epoch == 20:
            #     print(col_ind)
            #     print('-------')
            #     print(labels)
            pros_gt = labels.cpu()
            pros_gt = labels.long().tolist()
            pros = col_ind.tolist()

            index_gt1 = pros_gt.index(max(pros_gt))
            if pros_gt[index_gt1] == pros[index_gt1]:
                nums_first += 1
            del pros_gt[index_gt1]
            del pros[index_gt1]

            index_gt2 = pros_gt.index(max(pros_gt))
            if pros_gt[index_gt2] == pros[index_gt2]:
                nums_second += 1
            del pros_gt[index_gt2]
            del pros[index_gt2]

            index_gt3 = pros_gt.index(max(pros_gt))
            if pros_gt[index_gt3] == pros[index_gt3]:
                nums_third += 1
            del pros_gt[index_gt3]
            del pros[index_gt3]

            _, pred = torch.max(out, 1)
            #计算srcc plcc rmse
            test_srcc, _ = stats.spearmanr(col_ind, labels.cpu())
            test_plcc, _ = stats.pearsonr(col_ind, labels.cpu())
            test_rmse    = rmse( col_ind, labels.cpu() )

            if np.isnan(test_srcc) or np.isnan(test_plcc):
                test_srcc=0
                test_plcc=0
                testlen-=1


            eval_srcc += test_srcc
            eval_plcc += test_plcc
            eval_rmse += test_rmse
            col_ind = torch.tensor(col_ind)
            col_ind = col_ind.cuda()
            #计算准确个数
            num_correct = (col_ind == labels.long()).sum()
            eval_acc += num_correct.item()

            col_ind = col_ind.cpu().numpy()
            col_ind = col_ind.tolist()
            # save.extend(col_ind)
            txt_id = ii+1
            txt = txt_path + txts_path[ii]
            img = img_path + imgs_path[ii]
            img = cv2.imread(img)
            labels_unisal  = labels_set_unisal[(ii*5):(ii*5)+5]
            labels_eml  = labels_set_eml[(ii*5):(ii*5)+5]
            labels_salicon  = labels_set_salicon[(ii*5):(ii*5)+5]
            labels_salgan  = labels_set_salgan[(ii*5):(ii*5)+5]

            f = open(txt, "r")
            labels_id = 0
            while True:

                line = f.readline()
                if line:
                    pass  # do something here

                    label = labels[labels_id]
                    label_unisal = labels_unisal[labels_id]
                    label_eml = labels_eml[labels_id]
                    label_salicon = labels_salicon[labels_id]
                    label_salgan = labels_salgan[labels_id]

                    col   = col_ind[labels_id]
                    if label==col:
                        color = (0,255,0)
                    else:
                        color = (0,0,255)

                    if label == label_unisal:
                        color_unisal = (0, 255, 0)
                    else:
                        color_unisal = (0, 0, 255)

                    if label == label_eml:
                        color_eml = (0, 255, 0)
                    else:
                        color_eml = (0, 0, 255)

                    if label == label_salicon:
                        color_salicon = (0, 255, 0)
                    else:
                        color_salicon = (0, 0, 255)
                    if label == labels_salgan:
                        color_salgan = (0, 255, 0)
                    else:
                        color_salgan = (0, 0, 255)



                    line = line.strip()

                    line = line.split(',')
                    x1 = line[0]
                    y1 = line[1]
                    x2 = line[2]
                    y2 = line[3]
                    xcenter = line[4]
                    ycenter = line[5]
                    # print(ycenter)

                    x1 = int(float(x1))
                    y1 = int(float(y1))
                    x2 = int(float(x2))
                    y2 = int(float(y2))
                    xcenter = int(float(xcenter))
                    ycenter = int(float(ycenter))
                    text = str(col)
                    text_unisal = str(label_unisal)
                    text_eml = str(label_eml)
                    text_salgan = str(label_salgan)
                    text_salicon = str(label_salicon)
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                    img = cv2.putText(img, text, (xcenter, ycenter), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

                    img_unisal = cv2.rectangle(img, (x1, y1), (x2, y2), color_unisal, 4)
                    img_unisal = cv2.putText(img_unisal, text_unisal, (xcenter, ycenter), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

                    img_eml = cv2.rectangle(img, (x1, y1), (x2, y2), color_eml, 4)
                    img_eml = cv2.putText(img_eml, text_eml, (xcenter, ycenter), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

                    img_salgan = cv2.rectangle(img, (x1, y1), (x2, y2), color_salgan, 4)
                    img_salgan = cv2.putText(img_salgan, text_salgan, (xcenter, ycenter), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

                    img_salicon = cv2.rectangle(img, (x1, y1), (x2, y2), color_salicon, 4)
                    img_salicon = cv2.putText(img_salicon, text_salicon, (xcenter, ycenter), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

                    labels_id+=1

                else:
                    # print(int(test_srcc))
                    if test_srcc>0.9:

                        cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/可视化/salicon/mymodel/好/{:06d}.jpg'.format(txt_id), img)
                        cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/可视化/salicon/unisal/{:06d}.jpg'.format(txt_id), img_unisal)
                        cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/可视化/salicon/salicon/{:06d}.jpg'.format(txt_id), img_salicon)
                        cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/可视化/salicon/eml/{:06d}.jpg'.format(txt_id), img_eml)
                        cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/可视化/salicon/salgan/{:06d}.jpg'.format(txt_id), img_salgan)


                    if test_srcc<0.2:
                        cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/可视化/salicon/mymodel/坏/{:06d}.jpg'.format(txt_id), img)

                    break



        srcc=eval_srcc/testlen
        acc_first = nums_first / testlen
        acc_second = nums_second / testlen
        acc_third = nums_third / testlen
        # if acc_first >= best_acc_first:
        #     best_acc_first = acc_first
        #     best_epoch = epoch + 1
            # torch.save(model.state_dict(), '/home/w509/1workspace/lee/360_fix_sort/checkpoint/6classifer_100.pt')

        print('Test Loss: {:.6f}, Acc: {:.6f},Srcc: {:.6f},Plcc: {:.6f},Rmse: {:.6f},bestSrcc:{:.6f},ACC_first:{:04f},ACC_second:{:04f},ACC_third:{:04f},best_epoch:{}'.format(eval_loss / (len(
            testset)), eval_acc / (len(testset)), eval_srcc/testlen, eval_plcc/testlen, eval_rmse/testlen ,bestsrcc,acc_first,acc_second,acc_third, best_epoch))
    model.optimizer = optimizer
    total_time = timer() - overall_start
    # save = np.array(save)
    # f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/dut_new_select/pred_6classifier.h5', 'w')
    # # f.create_dataset('train', data=train_set)
    # f.create_dataset('labels', data=save)
    # f.close()


