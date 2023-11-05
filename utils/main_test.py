import os
import cv2
from cut_panoimg import cutOutPannoama
from cut_panoimg_global import cutOutPannoama_one

global image_id
image_id = 0
global image_id1
image_id1 = 0
xcenter_array = []
ycenter_array = []


path1 = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/360_new_dataset/test/testlocal/'
imgspath = r'/media/w509/967E3C5E7E3C3977/1workspace/dataset/360_new_dataset_select/salgan_sal/test/'



imgs_path1 = os.listdir(path1)
imgs_path1.sort(key=lambda x:int(x[:-4]))



for j in range(0, len(imgs_path1)):


    img_txt = imgs_path1[j]
    img_name = img_txt[:-4] + '.jpg'

    img_path = imgspath + '%s' % (img_name)


    img = cv2.imread(img_path)




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


            save_path = '/home/w509/1workspace/lee/360_fix_sort/box/360_new_dataset_ceshi/test_salgan/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = img_txt[:-4] + '_' + str('{:06d}'.format(image_id1)) + '.jpg'

            cv2.imwrite(save_path + save_name, img_box1)

            print("create %s {:06d}".format(j) % line)
        else:
            break
    f.close()


