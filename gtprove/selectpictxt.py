import os
from shutil import copy

nums = 250

imghdr ='/home/w509/1workspace/lee/360_fix_sort/可视化/salicon/mymodel/好/'
imgdir = '/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/2dfixdataset/images/val/'
imgtxt = '/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/boxtxt/vallocal/'
imgs_path = os.listdir(imghdr)
imgs_path.sort(key=lambda x:int(x[:-4]))
print(imgs_path)
# imgs_txt = os.listdir(imgtxt)
# imgs_txt.sort(key=lambda x:int(x[:-8]))
# print(imgs_txt)
tocopyimgpath = '/home/w509/1workspace/lee/360_fix_sort/gtprove/selectpic/'
tocopytxtpath = '/home/w509/1workspace/lee/360_fix_sort/gtprove/selecttxt/'
for num in range(0,nums):
    fromimgname = imgdir + imgs_path[num]
    fromtxtname = imgtxt + imgs_path[num] + '.txt'
    copy(fromtxtname,tocopytxtpath)
    copy(fromimgname, tocopyimgpath)