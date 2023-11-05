import cv2
import os
import numpy as np
import json
img_path = '/home/w509/1workspace/lee/360_fix_sort/gtprove/selectpic_drawlabel/'
txt_path = '/home/w509/1workspace/lee/360_fix_sort/gtprove/selecttxt/'
json_path = '/home/w509/1workspace/lee/360_fix_sort/gtprove/json/fixation_val.json'
start_nums = 1

imgs_path = os.listdir(img_path)
txts_path = os.listdir(txt_path)
imgs_path.sort(key=lambda x: int(x[:-4]))

txts_path.sort(key=lambda x: int(x[:-8]))

if start_nums ==1:
    all_label_list = []
else:
    all_label_list = json.load(open(json_path, 'r'))








for ii in range(start_nums-1,len(imgs_path)):
    txt = txt_path + txts_path[ii]
    img = img_path + imgs_path[ii]

    imgs = cv2.imread(img)

    print('第{}张图片'.format(ii+1))
    f = open(txt, "r")
    labels_id = 0
    pro_nums = 0
    while True:

        line = f.readline()
        if line:
            pass  # do something here
            pro_nums+=1
        else:


            break
    f.close()
    print('本图片共有{}个物体'.format(pro_nums))
    label_list = []
    pro_labels = []
    paixu_nums = 9
    for i in range(0,pro_nums):
        label_list.append(0)
    label_numpy = np.array(label_list)
    f = open(txt, "r")
    while True:

        line = f.readline()
        if line:
            pass  # do something here

            cv2.namedWindow("do not click x!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            cv2.imshow("do not click x!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", imgs)
            # cv2.resizeWindow('image',1280,640)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            pro_label = input('请输入最显著的物体编号:')
            if len(pro_label) == 0:
                pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

            if len(pro_label) == 0:
                pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

            if len(pro_label) == 0:
                pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

            if len(pro_label) == 0:
                pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

            if len(pro_label) == 0:
                pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

            # pro_label = int(pro_label)
            flag=1
            for la in pro_labels:
                if la==int(pro_label):
                    flag=0
            if flag==0:
                pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                flag = 1
                for la in pro_labels:
                    if la==int(pro_label):
                        flag = 0
                if flag == 0:
                    pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    flag = 1
                    for la in pro_labels:
                        if la==int(pro_label):
                            flag = 0
                if flag == 0:
                    pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    flag = 1
                    for la in pro_labels:
                        if la==int(pro_label):
                            flag = 0
                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')


            pro_label = int(pro_label)
            if pro_label>pro_nums:
                pro_label = input('错误，超过范围，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')


                flag = 1
                for la in pro_labels:
                    if la==int(pro_label):
                        flag = 0
                if flag == 0:
                    pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    flag = 1
                    for la in pro_labels:
                        if la==int(pro_label):
                            flag = 0
                    if flag == 0:
                        pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        flag = 1
                        for la in pro_labels:
                            if la==int(pro_label):
                                flag = 0
                    if flag == 0:
                        pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        flag = 1
                        for la in pro_labels:
                            if la==int(pro_label):
                                flag = 0
                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

            pro_label = int(pro_label)
            if pro_label>pro_nums:
                pro_label = input('错误，超过范围，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')


                flag = 1
                for la in pro_labels:
                    if la==int(pro_label):
                        flag = 0
                if flag == 0:
                    pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    flag = 1
                    for la in pro_labels:
                        if la==int(pro_label):
                            flag = 0
                    if flag == 0:
                        pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        flag = 1
                        for la in pro_labels:
                            if la==int(pro_label):
                                flag = 0
                    if flag == 0:
                        pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        flag = 1
                        for la in pro_labels:
                            if la==int(pro_label):
                                flag = 0
                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

            pro_label = int(pro_label)
            if pro_label>pro_nums:
                pro_label = input('错误，超过范围，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                if len(pro_label) == 0:
                    pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')


                flag = 1
                for la in pro_labels:
                    if la==int(pro_label):
                        flag = 0
                if flag == 0:
                    pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    flag = 1
                    for la in pro_labels:
                        if la==int(pro_label):
                            flag = 0

                    if flag == 0:
                        pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        flag = 1
                        for la in pro_labels:
                            if la==int(pro_label):
                                flag = 0
                    if flag == 0:
                        pro_label = input('错误，输入重复，请重新输入最显著的物体编号:')
                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        if len(pro_label) == 0:
                            pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                        flag = 1
                        for la in pro_labels:
                            if la == int(pro_label):
                                flag = 0
                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

                    if len(pro_label) == 0:
                        pro_label = input('错误,输入为空，请重新输入最显著的物体编号:')

            pro_label = int(pro_label)
            label_numpy[pro_label-1]=paixu_nums
            paixu_nums-=1
            pro_labels.append(pro_label)


        else:
            label_list = label_numpy.tolist()
            all_label_list.append(label_list)

            if os.path.exists(json_path):
                os.remove(json_path)
            with open(json_path, 'w') as f1:
                json.dump(all_label_list, f1)
            f1.close()

            break
    f.close()

print('共标注了{}张图片'.format(len(all_label_list)))


# print()
