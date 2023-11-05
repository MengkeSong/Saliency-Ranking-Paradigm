import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.optimize import linear_sum_assignment
from model.net4multiclassi_unisal import VisionTransformer
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
import shutil

import seaborn as sns
import pandas as pd
# from  net import GCN_Net,PyGCN
import model
# from model.net4multiclassi_unisal import GCN_Net
from torch.optim import *
import os

print(os.listdir("../"))
if __name__ == "__main__":

    EPOCHS = 2000
    BATCH_SIZE = 5
    learning_rate = 0.001
    import torch.utils.data as data
    import torch
    import h5py




    class DataFromH5File(data.Dataset):
        def __init__(self, filepath):
            h5File = h5py.File(filepath, 'r')
            self.data1 = h5File['train1']
            self.data2 = h5File['train2']
            self.data3 = h5File['train3']
            self.xcenter = h5File['xcenter']
            self.ycenter = h5File['ycenter']

            self.labels = h5File['labels']

        def __getitem__(self, idx):
            label = torch.tensor(self.labels[idx]).float()
            data1 = torch.tensor(self.data1[idx]).float()
            data2 = torch.tensor(self.data2[idx]).float()
            data3 = torch.tensor(self.data3[idx]).float()
            xcenter = torch.tensor(self.xcenter[idx]).float()
            ycenter = torch.tensor(self.ycenter[idx]).float()


            return data1,data2,data3,xcenter,ycenter, label

        def __len__(self):
            assert self.data1.shape[0] == self.labels.shape[0], "Wrong data length"
            return self.data1.shape[0]


    class DatatestFromH5File(data.Dataset):
        def __init__(self, filepath):
            h5File = h5py.File(filepath, 'r')
            self.data1 = h5File['train1']
            self.data2 = h5File['train2']
            self.data3 = h5File['train3']
            self.xcenter = h5File['xcenter']
            self.ycenter = h5File['ycenter']
            self.labels = h5File['labels']

        def __getitem__(self, idx):
            label = torch.tensor(self.labels[idx]).float()
            data1 = torch.tensor(self.data1[idx]).float()
            data2 = torch.tensor(self.data2[idx]).float()
            data3 = torch.tensor(self.data3[idx]).float()
            xcenter = torch.tensor(self.xcenter[idx]).float()
            ycenter = torch.tensor(self.ycenter[idx]).float()


            return data1,data2,data3,xcenter,ycenter, label

        def __len__(self):
            assert self.data1.shape[0] == self.labels.shape[0], "Wrong data length"
            return self.data1.shape[0]


    def isnull(list):
        gt_id = 0
        for gt in list:
            if gt > 0:
                gt_id = 1
        return gt_id


    testset = DatatestFromH5File(
        "/home/w509/1workspace/lee/360_fix_sort/feature/ranking_unisal_resnet50_feature/val_select_local_global_unisal_multi_classi_0_5-0_75.h5")
    test_loader = data.DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = VisionTransformer()

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
    model.load_state_dict(torch.load('/home/w509/1workspace/lee/360_fix_sort/runs/runs_datacate_unisal_resnet/checkpoint/model_transformer_train_01.pt'))
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

    test_image_num=0

    best_acc_first = 0
    with torch.no_grad():
        model.eval()  # 模型评估
        eval_loss = 0
        eval_acc = 0
        eval_f1 = 0
        eval_precsion = 0
        eval_recall = 0
        eval_srcc = 0
        eval_plcc = 0
        eval_rmse = 0
        testlen_first = 0
        testlen_second = 0
        testlen_third = 0
        # testlen = (len(testset) / 5)

        nums_first = 0
        nums_second = 0
        nums_third = 0
        # path = r'//home/w509/1workspace/lee/360_fix_sort/boxtxt/sitzmann/test/'
        # imgs_path = os.listdir(path)
        # imgs_path.sort(key=lambda x: int(x[:-4]))
        for ii, (data1,data2,data3,xcenter,ycenter, labels) in enumerate(test_loader):  # 测试模型
            if CUDA:
                data1, data2,data3, xcenter, ycenter, labels = data1.cuda(), data2.cuda(),data3.cuda(), xcenter.cuda(), ycenter.cuda(), labels.cuda()

            optimizer.zero_grad()
            torch.cuda.empty_cache()





            out = model(data1,data2,data3,xcenter,ycenter)

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



            _, pred = torch.max(out, 1)
            #计算srcc plcc rmse
            test_srcc, _ = stats.spearmanr(col_ind, labels.cpu())
            test_plcc, _ = stats.pearsonr(col_ind, labels.cpu())
            test_rmse    = rmse( col_ind, labels.cpu() )

            test_f1 = f1_score(col_ind, labels.cpu(), average='macro')
            test_p = precision_score(col_ind, labels.cpu(), average='macro')
            test_r = recall_score(col_ind, labels.cpu(), average='macro')




            # test_srcc, _ = stats.spearmanr(pred.cpu(), labels.cpu())
            # test_plcc, _ = stats.pearsonr(pred.cpu(), labels.cpu())
            # test_rmse    = rmse(pred.cpu(), labels.cpu() )

            if np.isnan(test_srcc) or np.isnan(test_plcc):
                test_srcc=0
                test_plcc=0
                testlen-=1



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
            # shutil.copy(img_path,dest_img_path)
            # shutil.copy(txt_path,dest_txt_path)
 

            eval_srcc += test_srcc
            eval_plcc += test_plcc
            eval_rmse += test_rmse
            eval_recall += test_r
            eval_f1 += test_f1
            eval_precsion += test_p
            test_image_num+=1


            testlen=test_image_num


            col_ind = torch.tensor(col_ind)
            col_ind = col_ind.cuda()
            #计算准确个数
            # num_correct = (col_ind == labels.long()).sum()
            num_correct = (pred == labels.long()).sum()

            eval_acc += num_correct.item()

        srcc=eval_srcc/testlen
        acc_first = nums_first / testlen
        acc_second = nums_second / testlen
        acc_third = nums_third / testlen
        if srcc >= best_acc_first:
            best_acc_first = srcc

            # torch.save(model.state_dict(), '/home/w509/1workspace/lee/360_fix_sort/runs/runs_datacate_unisal_resnet/checkpoint/model_transformer.pt')

        print('Test Loss: {:.6f}, Acc: {:.6f},Srcc: {:.6f},Plcc: {:.6f},Rmse: {:.6f},,precision: {:.6f},,recall: {:.6f},,f1: {:.6f},bestSrcc:{:.6f},ACC_first:{:04f},ACC_second:{:04f},ACC_third:{:04f},best_epoch:{}'.format(eval_loss / (len(
            testset)), eval_acc / (len(testset)), eval_srcc/testlen, eval_plcc/testlen, eval_rmse/testlen, eval_precsion/testlen, eval_recall/testlen, eval_f1/testlen ,bestsrcc,acc_first,acc_second,acc_third, best_epoch))
    model.optimizer = optimizer
    print(test_image_num)
    total_time = timer() - overall_start


