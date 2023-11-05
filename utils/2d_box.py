import os
import cv2

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

path = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/sitzmann/trainlocal/'
path1 = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/sitzmann/trainglobal/'

imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x:int(x[:-4]))
print(imgs_path)

imgs_path1 = os.listdir(path1)
imgs_path1.sort(key=lambda x:int(x[:-4]))



for j in range(0, len(imgs_path)):


    img_txt = imgs_path[j]
    img_name = img_txt[:-4] + '.jpg'
    gt_name = img_txt[:-4]
    img_path = r'/media/w509/967E3C5E7E3C3977/1workspace/dataset/sitzmann/images/' + '%s' % (img_name)
    # gt_path  = r'/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/2dfixdataset/maps/train/' + '%s' % (gt_name) + '.png'
    img = cv2.imread(img_path)
    img_width = img.shape[1]
    resize_width = img_width
    img_height = img.shape[0]
    resize_height = img_height
    # gt  = cv2.imread(gt_path)
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
            xcenter = int(float(xcenter))
            xcenter = xcenter/resize_width
            xcenter = int(xcenter)

            ycenter = int(float(ycenter))
            ycenter = ycenter/resize_height
            ycenter = int(ycenter)


            xcenter_array.append(xcenter)
            ycenter_array.append(ycenter)

            img_box = img[y1:y2, x1:x2]
            # gt_box  = gt[y1:y2, x1:x2]

            save_path = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/box/sitzmann/train_box_local/'
            # save_gtpath = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/box/salicon_rand_max20/train_box_gt/'
            xycenter_path = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/sitzmann/sitzmann/xycenter'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if not os.path.exists(xycenter_path):
                os.makedirs(xycenter_path)

            # if not os.path.exists(save_gtpath):
            #     os.makedirs(save_gtpath)

            save_name = img_txt[:-4] + '_' + str('{:06d}'.format(image_id)) + '.jpg'

            cv2.imwrite(save_path + save_name, img_box)
            # cv2.imwrite(save_gtpath + save_name,gt_box)

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
            filename = line[0:p]

            image_id1 += 1

            line = filename.split(',')
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]


            x1 = int(float(x1))
            y1 = int(float(y1))
            x2 = int(float(x2))
            y2 = int(float(y2))

            img_box = img[y1:y2, x1:x2]

            save_path = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/box/sitzmann/train_box_global/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = img_txt[:-4] + '_' + str('{:06d}'.format(image_id1)) + '.jpg'

            cv2.imwrite(save_path + save_name, img_box)

            print("create %s {:06d}".format(j) % line)
        else:
            break
    f.close()
import numpy as np

xcenter_array=np.array(xcenter_array)
ycenter_array=np.array(ycenter_array)

print(xcenter_array.shape)
print(ycenter_array.shape)
import h5py
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/sitzmann/sitzmann/xycenter/xcenter_train.h5', 'w')
f.create_dataset('xcenter', data=xcenter_array)
f.close()
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/sitzmann/sitzmann/xycenter/ycenter_train.h5', 'w')
f.create_dataset('ycenter', data=ycenter_array)
f.close()









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

