import os
import cv2
from cut_panoimg import cutOutPannoama
from PIL import Image
import h5py
import numpy as np

global image_id
image_id = 0
global image_id1
image_id1 = 0
xcenter_array = []
ycenter_array = []

# path = r'/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/box/salicon_rand/val_box_local/'
# path1 = r'/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/box/salicon_rand/val_box_global/'
# path2 = r'/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/box/salicon_rand/val_box_sal_salgan/'
# path3 = r'/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/box/salicon_rand/val_box_gt/'
# path2 = r'/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/salient360_img/box/val_box_gt/'
pathtxt1 = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/salicon_rand/trainlocal/'
pathtxt2 = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/salicon_rand/trainglobal/'

txtpath1 = os.listdir(pathtxt1)
txtpath2 = os.listdir(pathtxt2)
txtpath1.sort(key=lambda x:str(x[:-4]))
txtpath2.sort(key=lambda x:str(x[:-4]))


# imgs_path = os.listdir(path)
# imgs_path.sort(key=lambda x:str(x[:-4]))
# print(imgs_path)
#
# imgs_path1 = os.listdir(path1)
# imgs_path1.sort(key=lambda x:str(x[:-4]))
#
# imgs_path2 = os.listdir(path2)
# imgs_path2.sort(key=lambda x:str(x[:-4]))
#
# imgs_path3 = os.listdir(path3)
# imgs_path3.sort(key=lambda x:str(x[:-4]))

# imgs_path2 = os.listdir(path2)
# imgs_path2.sort(key=lambda x:str(x[:-4]))
# img_local = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/randmbox/salicon_rand/val_local'
# img_global = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/randmbox/salicon_rand/val_global'
# img_sal = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/randmbox/salicon_rand/val_sal'
# img_flo = '/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/randmbox/salicon_rand/val_gt'
# save_txt ='/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/boxtxt/salicon_multi_rand/vallocal'
# img_gt = '/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/salient360_img/cut_pano/randmbox_noprojection/val_box_gt'
# if not os.path.exists(img_local):
#     os.makedirs(img_local)
# if not os.path.exists(img_global):
#     os.makedirs(img_global)
# if not os.path.exists(img_sal):
#     os.makedirs(img_sal)
# if not os.path.exists(img_flo):
#     os.makedirs(img_flo)
# if not os.path.exists(img_gt):
#     os.makedirs(img_gt)
global ids
ids = 1
global ids_global
ids_global = 1

global ids_gt
ids_gt = 1
j=0



gt_dataset = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_rand/h5_blcakdot_nums/multiclassi_labels/train_multiclassi_0_4.h5', 'r')
gt_dataset = np.array(gt_dataset['labels'][:])

gt_set = []


for i in range(0,len(txtpath1)):
    img_txt = txtpath1[i]
    print(img_txt)
    txt_path = pathtxt1 + '/' + img_txt
    f = open(txt_path, "r")
    prolen = 0
    while True:

        line = f.readline()
        if line:
            prolen+=1
        else:
            break
    f.close()
    # f1 = open(save_txt+ '/' + img_txt,'a')
    #
    # img_l = imgs_path[j:j+prolen]
    # img_g = imgs_path1[j:j+prolen]
    # img_s = imgs_path2[j:j+prolen]
    # img_f = imgs_path3[j:j+prolen]
    gt_labels = gt_dataset [j:j+prolen]
    j+=prolen
    # img_t = imgs_path2[j:j+10]

    # if prolen==10:
    #
    #
    #
    #     for t in range(0,3):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    #
    #     for t in range(1,4):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    #
    #     for t in range(2,5):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #
    #         gt_set.append(gt_labels[t])
    #
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    #
    #     for t in range(3,6):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    #     for t in range(4,7):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    #     for t in range(5,8):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    #
    #     for t in range(6,9):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #         gt_set.append(gt_labels[t])
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    #
    #     for t in range(7,10):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #
    #         gt_set.append(gt_labels[t])
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    #
    #     for t in [8,9,0]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #
    #         gt_set.append(gt_labels[t])
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    #
    #     for t in [9,0,1]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) +'.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) +'.jpg')
    #
    #         gt_set.append(gt_labels[t])
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e ==t:
    #                 f1.write('%s' % (line) )
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids +=1
    # if prolen==9:
    #     for t in range(0, 3):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #
    #         gt_set.append(gt_labels[t])
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(1, 4):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(2, 5):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(3, 6):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #
    #         gt_set.append(gt_labels[t])
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #     for t in range(4, 7):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(5,8):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(6,9):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in [7, 8, 0]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #     for t in [8, 0, 1]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #
    #     # for t in range(0, 5):
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global)+ '.jpg')
    #     #     ids_global += 1
    #     #
    #     # for t in range(1, 6):
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global)+ '.jpg')
    #     #     ids_global += 1
    #     #
    #     # for t in range(2, 7):
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global)+ '.jpg')
    #     #     ids_global += 1
    #     #
    #     # for t in range(3, 8):
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global)+ '.jpg')
    #     #     ids_global += 1
    #     # for t in range(4, 9):
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global) + '.jpg')
    #     #     ids_global += 1
    #     # for t in range(5, 10):
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global)+ '.jpg')
    #     #     ids_global += 1
    #     #
    #     # for t in [6, 7, 8, 9, 1]:
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global) + '.jpg')
    #     #     ids_global += 1
    #     #
    #     # for t in [7, 8, 9, 1, 2]:
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global)+ '.jpg')
    #     #     ids_global += 1
    #     #
    #     # for t in [8, 9, 1, 2, 3]:
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global)+ '.jpg')
    #     #     ids_global += 1
    #     #
    #     # for t in [9, 1, 2, 3, 4]:
    #     #     im = Image.open(path1 + '/' + img_g[t])
    #     #     im.save(img_global + '/' + '{:06d}'.format(ids_global)+ '.jpg')
    #     #     ids_global += 1
    # if prolen==8:
    #     for t in range(0, 3):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(1, 4):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(2, 5):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(3, 6):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #     for t in range(4,7):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(5,8):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in [6, 7, 0]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in [7, 0, 1]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    # if prolen==7:
    #     for t in range(0, 3):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(1, 4):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(2, 5):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(3,6):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #     for t in range(4,7):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in [5, 6, 0]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in [6, 0, 1]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #
    #         gt_set.append(gt_labels[t])
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    # if prolen==6:
    #     for t in range(0, 3):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(1, 4):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(2,5):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in range(3,6):
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #     for t in [4, 5, 0]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    #
    #     for t in [5, 0, 1]:
    #         e = 0
    #         im_local = Image.open(path + '/' + img_l[t])
    #         im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_global = Image.open(path1 + '/' + img_g[t])
    #         im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_sal = Image.open(path2 + '/' + img_s[t])
    #         im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
    #         im_flo = Image.open(path3 + '/' + img_f[t])
    #         im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
    #         gt_set.append(gt_labels[t])
    #
    #
    #         f2 = open(txt_path, "r")
    #         while True:
    #
    #             line = f2.readline()
    #             if line and e == t:
    #                 f1.write('%s' % (line))
    #             else:
    #
    #                 if not line:
    #                     break
    #             e += 1
    #         f2.close()
    #         ids += 1
    if prolen==5:
        for t in range(0, 3):
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])


            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
        for t in range(1, 4):
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])


            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
        for t in range(2, 5):
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])


            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
        for t in [3,4,0]:
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])

            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
        for t in [4,0,1]:
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])

            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
    if prolen==4:
        for t in range(0,3):
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])


            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
        for t in range(1,4):
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])

            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
        for t in [2,3,0]:
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])


            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
        for t in [3,0,1]:
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])


            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
    if prolen==3:
        for t in range(0,3):
            # e = 0
            # im_local = Image.open(path + '/' + img_l[t])
            # im_local.save(img_local + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_global = Image.open(path1 + '/' + img_g[t])
            # im_global.save(img_global + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_sal = Image.open(path2 + '/' + img_s[t])
            # im_sal.save(img_sal + '/' + '{:06d}'.format(ids) + '.jpg')
            # im_flo = Image.open(path3 + '/' + img_f[t])
            # im_flo.save(img_flo + '/' + '{:06d}'.format(ids) + '.jpg')
            gt_set.append(gt_labels[t])


            # f2 = open(txt_path, "r")
            # while True:
            #
            #     line = f2.readline()
            #     if line and e == t:
            #         f1.write('%s' % (line))
            #     else:
            #
            #         if not line:
            #             break
            #     e += 1
            # f2.close()
            # ids += 1
    # f1.close()





gt_set_np=np.array(gt_set)
print(gt_set_np.shape)
f = h5py.File('/media/w509/967E3C5E7E3C3977/1workspace/code/360_fix_sort/feature/salicon_multi_rand/h5_blackdot_nums/multiclassi_labels/train_multiclassi_0_4.h5', 'w')

f.create_dataset('labels', data=gt_set_np)
f.close()












