import cv2
import numpy as np
from PIL import Image
import os
import h5py

img_path = '/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/2dfixdataset/salmap_emlnet/res50_maps/'  # 输入文件夹地址

gt_name = os.listdir(img_path)
gt_name.sort(key=lambda x: int(x[-16:-4]))

for i in range(len(gt_name)):
    img_name = img_path+gt_name[i]
    change_img_name = img_path + '{:06d}'.format(i+1)+'.jpg'
    os.rename(img_name,change_img_name)
