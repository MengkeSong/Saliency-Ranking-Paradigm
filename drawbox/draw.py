import cv2
txt_path='/home/w509/1workspace/lee/360_fix_sort/drawbox/000826.jpg.txt'
img = '/home/w509/1workspace/lee/360_fix_sort/drawbox/000826.jpg'
img_pic = '/home/w509/1workspace/lee/360_fix_sort/drawbox/000826_pic.jpg'

img = cv2.imread(img)
img_pic = cv2.imread(img_pic)
f = open(txt_path, "r")
while True:
    line = f.readline()
    if line:
        pass  # do something here
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

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        img_pic = cv2.rectangle(img_pic, (x1, y1), (x2, y2), (0,0,255), 4)

    else:
        cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/drawbox/1.jpg',img)
        cv2.imwrite('/home/w509/1workspace/lee/360_fix_sort/drawbox/1_pic.jpg',img_pic)
        break