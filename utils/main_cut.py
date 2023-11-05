import os
import cv2
from cut_panoimg import cutOutPannoama
from cut_panoimg_global import cutOutPannoama_one

global image_id
image_id = 0

xcenter_array = []
ycenter_array = []

path = r'/home/w509/1workspace/lee/360_fix_sort/boxtxt/360_new_dataset/train/trainlocal/'

imgspath = r'/home/w509/1workspace/lee/360_fix_sort/trans/train_box_sal1/'

imgs_path = os.listdir(path)
imgs_path.sort(key=lambda x:int(x[:-4]))
print(imgs_path)

sal_imgs = os.listdir(imgspath)
sal_imgs.sort(key=lambda x:int(x[:-4]))





for j in range(0, len(imgs_path)):


    img_txt = imgs_path[j]
    img_name = img_txt[:-4]




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

            img_path = imgspath  + '%s' % (img_name)+'_' + '{:06d}'.format(image_id) + '.jpg'

            img_box = cv2.imread(img_path)





            img_box1 = img_box[y1:y2,x1:x2]





            save_path = '/home/w509/1workspace/lee/360_fix_sort/box/360_new_dataset/train_box_sal/'


            if not os.path.exists(save_path):
                os.makedirs(save_path)



            # if not os.path.exists(save_salpath):
            #     os.makedirs(save_salpath)

            save_name = img_txt[:-4] + '_' + str('{:06d}'.format(image_id)) + '.jpg'

            cv2.imwrite(save_path + save_name, img_box1)

            # cv2.imwrite(save_salpath + save_name,sal_box1)



            print("create %s {:06d}".format(j) % line)
        else:
            break
    f.close()













