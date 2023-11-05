import os
import cv2
from cut_panoimg import cutOutPannoama
from cut_panoimg_global import cutOutPannoama_one
# davis_train
# davis_test
# Visal
# Segtrack-v2
# Easy-35
# VOS_test
global image_id
image_id = 0
global image_id1
image_id1 = 0
xcenter_array = []
ycenter_array = []

path = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/360_new_dataset/test/testlocal/'
path1 = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/360_new_dataset/test/testglobal/'
imgspath = r'/media/w509/967E3C5E7E3C3977/1workspace/dataset/360_new_dataset_select/images/test/'
gtspath =  r'/media/w509/967E3C5E7E3C3977/1workspace/dataset/360_new_dataset_select/gt/test/'
flospath = r'/media/w509/967E3C5E7E3C3977/1workspace/dataset/360_new_dataset_select/flo/test/'
# salmappath=r'/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/360_saliency_select/outputsal/'
imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x:int(x[:-4]))
print(imgs_path)

imgs_path1 = os.listdir(path1)
imgs_path1.sort(key=lambda x:int(x[:-4]))



for j in range(0, len(imgs_path)):


    img_txt = imgs_path[j]
    img_name = img_txt[:-4] + '.jpg'
    gt_name = img_txt[:-4]
    img_path = imgspath + '%s' % (img_name)
    gt_path  = gtspath + '%s' % (gt_name) + '.png'
    flo_path = flospath + '%s' %(img_name)
    # sal_path = salmappath + '%s' %(img_name)
    img = cv2.imread(img_path)
    gt  = cv2.imread(gt_path)
    flo = cv2.imread(flo_path)
    # sal = cv2.imread(sal_path)
    txt_path = path + '/' + img_txt

    f = open(txt_path, "r")
    while True:
        line = f.readline()
        if line:
            pass  # do something here
            line = line.strip()



            image_id +=1



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
            gt_box = gt[y1:y2, x1:x2]

            a = (x2 - x1)/2
            b = (y2 - y1)/2

            x1 = 250 - a
            x2 = 250 + a
            y1 = 250 - b
            y2 = 250 + b

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)




            xcenter = int(float(xcenter))
            ycenter = int(float(ycenter))

            xcenter_array.append(xcenter)
            ycenter_array.append(ycenter)

            img_box,flo_box = cutOutPannoama(img_path,flo_path,80,xcenter,ycenter)
            img_box1 = img_box[y1:y2,x1:x2]
            flo_box1 = flo_box[y1:y2, x1:x2]

            # gt_box  = cutOutPannoama(gt_path,25,xcenter,ycenter)

            # flo_box = cutOutPannoama(flo_path,80,xcenter,ycenter)
            #
            #
            # sal_box = cutOutPannoama(sal_path,80,xcenter,ycenter)


            save_path = '/home/w509/1workspace/lee/360_fix_sort/box/360_new_dataset/test_box_local/'
            save_gtpath = '/home/w509/1workspace/lee/360_fix_sort/box/360_new_dataset/test_box_gt/'
            save_flopath = '/home/w509/1workspace/lee/360_fix_sort/box/360_new_dataset/test_box_flo/'
            # save_salpath = '/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/box/360saliency_select_not10/box/val_box_sal/'



            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if not os.path.exists(save_gtpath):
                os.makedirs(save_gtpath)

            if not os.path.exists(save_flopath):
                os.makedirs(save_flopath)

            # if not os.path.exists(save_salpath):
            #     os.makedirs(save_salpath)

            save_name = img_txt[:-4] + '_' + str('{:06d}'.format(image_id)) + '.jpg'

            cv2.imwrite(save_path + save_name, img_box1)
            cv2.imwrite(save_gtpath + save_name,gt_box)
            cv2.imwrite(save_flopath + save_name,flo_box1)
            # cv2.imwrite(save_salpath + save_name,sal_box1)



            print("create %s {:06d}".format(j) % line)
        else:
            break
    f.close()

    txt_path1 = path1 + '/' + img_txt

    f = open(txt_path1, "r")
    while True:
        line = f.readline()
        if line:
            pass  # do something here
            line = line.strip()
            p = line.rfind('.')
            # filename = line[0:p]

            image_id1 += 1

            line = line.split(',')
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            xcenter = line[4]
            ycenter = line[5]


            x1 = int(float(x1))
            y1 = int(float(y1))
            x2 = int(float(x2))
            y2 = int(float(y2))

            a = (x2 - x1) / 2
            b = (y2 - y1) / 2

            x1 = 250 - a
            x2 = 250 + a
            y1 = 250 - b
            y2 = 250 + b

            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)


            xcenter = int(float(xcenter))
            ycenter = int(float(ycenter))

            img_box = cutOutPannoama_one(img_path,80,xcenter,ycenter)
            img_box1 = img_box[y1:y2, x1:x2]


            save_path = '/home/w509/1workspace/lee/360_fix_sort/box/360_new_dataset/test_box_global/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = img_txt[:-4] + '_' + str('{:06d}'.format(image_id1)) + '.jpg'

            cv2.imwrite(save_path + save_name, img_box1)

            print("create %s {:06d}".format(j) % line)
        else:
            break
    f.close()
import numpy as np

xcenter_array=np.array(xcenter_array)
ycenter_array=np.array(ycenter_array)

print(xcenter_array.shape)
print(ycenter_array.shape)
# import h5py
# f = h5py.File('/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/salient360_img/cut_pano/xycenter/xcenter_val.h5', 'w')
# f.create_dataset('xcenter', data=xcenter_array)
#
# f = h5py.File('/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/salient360_img/cut_pano/xycenter/ycenter_val.h5', 'w')
# f.create_dataset('ycenter', data=ycenter_array)









    # txt = open(txt_path, 'r')
    # file = txt.readline()
    # print(file)
    # file_txt = file

    # line = file_txt.split(',')
    #
    # x1 = line[1]
    # y1 = line[2]
    # x2 = line[3]
    # y2 = line[4]
    #
    #
    #
    # img_box = img[y1:y2, x1:x2]
    #
    # save_path = '/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/2dfixdataset/box/train_box_local/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # save_name = img_txt[:-4] + '_' + str(q) + '.jpg'
    #
    # cv2.imwrite(save_path + save_name, flo_box)

