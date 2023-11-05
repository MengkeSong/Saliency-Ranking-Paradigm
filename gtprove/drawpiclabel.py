import cv2
import os
txt='/home/w509/1workspace/lee/360_fix_sort/gtprove/selecttxt/'
img = '/home/w509/1workspace/lee/360_fix_sort/gtprove/selectpic/'
save_path = '/home/w509/1workspace/lee/360_fix_sort/gtprove/selectpic_drawlabel/'
imgs_path = os.listdir(img)
imgs_path.sort(key=lambda x:int(x[:-4]))
txts_path = os.listdir(txt)
txts_path.sort(key=lambda x:int(x[:-8]))

for i in range(len(imgs_path)):
    imgname = img + imgs_path[i]
    txtname = txt + txts_path[i]


    img1 = cv2.imread(imgname)
    label = 0
    f = open(txtname, "r")
    while True:

        line = f.readline()
        if line:
            pass  # do something here
            label+=1
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
            label=str(label)
            xcenter=int(xcenter)
            ycenter=int(ycenter)
            img1 = cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 4)
            img1 = cv2.putText(img1, label, (xcenter, ycenter), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
            label=int(label)

        else:
            save_name = save_path+imgs_path[i]
            cv2.imwrite(save_name,img1)

            break