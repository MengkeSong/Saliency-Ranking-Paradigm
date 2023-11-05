import cv2
# from test import cutPannoama
# from test import boderPannoama
# from test import add_border
import numpy as np
import cv2
from math import pi
from math import cos
from math import sin
from math import tan
from math import acos
from cmath import sqrt
from PIL import Image, ImageOps

def cutOutPannoama(im, flo, fov, cx, cy,panoOutWidth = 500):
    panoData = cv2.imread(im)
    copyData = panoData
    panofloData = cv2.imread(flo)
    # panosalData = cv2.imread(sal)
    sp = panoData.shape
    panoWidth = sp[1]
    panoHeight = sp[0]
    outWidth = panoOutWidth
    outHeight = panoOutWidth
    fov = fov * pi / 180
    ix = cx
    iy = cy
    plane1 = np.zeros((outHeight, outWidth, 3))
    plane2 = np.zeros((outHeight, outWidth, 3))
    # plane3 = np.zeros((outHeight, outWidth, 3))

    bigplane = np.zeros((panoHeight, panoWidth, 3))
    #     求焦距
    f = outWidth / (2 * tan(fov / 2))

    anglex = ix * 2 * pi / panoWidth
    angley = pi / 2 - iy * pi / panoHeight

    rotz = np.eye(3, dtype=float)
    roty = np.eye(3, dtype=float)
    # rot = np.eye(3, dtype=float)
    # lin=cos(anglex)
    rotz[0, 0] = cos(anglex)
    rotz[0, 1] = sin(anglex)
    rotz[1, 0] = -sin(anglex)
    rotz[1, 1] = cos(anglex)
    roty[0, 0] = cos(angley)
    roty[0, 2] = sin(angley)
    roty[2, 0] = -sin(angley)
    roty[2, 2] = cos(angley)
    # rot = roty * rotz

    rot = np.dot(roty, rotz)
    # rot = np.transpose(rot)
    rot = rot.T
    imgData = plane1
    floData = plane2
    # salData = plane3

    for i in range(0, outHeight):
        # 球面坐标系坐标
        tz = i - outHeight / 2
        for j in range(0, outWidth):
            tx = f
            ty = j - outWidth / 2

            x = rot[0, 0] * tx + rot[0, 1] * ty + rot[0, 2] * tz
            y = rot[1, 0] * tx + rot[1, 1] * ty + rot[1, 2] * tz
            z = rot[2, 0] * tx + rot[2, 1] * ty + rot[2, 2] * tz
            # 转换成球坐标
            theta = acos(z / sqrt(x * x + y * y + z * z))
            fi = acos(x / sqrt(x * x + y * y))
            if (y < 0):
                fi = 2 * pi - fi
            # 求全景图坐标
            i2 = int(abs(theta * panoHeight / pi))
            j2 = int(abs(fi * panoWidth / (2 * pi)))
            # print(i2,j2)

            if (i2 < 0 and i2 >= panoHeight and j2 < 0 and j2 >= panoWidth):
                continue

            for k in range(0, 3):
                imgData[outHeight - i - 1, j, k] = panoData[i2-1, j2-1, k]
                floData[outHeight - i - 1, j, k] = panofloData[i2 - 1, j2 - 1, k]
                # salData[outHeight - i - 1, j, k] = panosalData[i2 - 1, j2 - 1, k]
                # bigplane[i2, j2, k] = panoData[i2, j2, k]
                # copyData[i2, j2, k] = panoData[i2, j2, k]

    imgData = imgData.astype(np.uint8)
    floData = floData.astype(np.uint8)
    # salData = salData.astype(np.uint8)

    # bigplane = bigplane.astype(np.uint8)

    return imgData , floData
if __name__ == "__main__":
    im='/home/w509/1workspace/lee/Yet-Another-EfficientDet-Pytorch-master/360_saliency_select/videos/000004.jpg'

    cutoutimg=cutOutPannoama(im, 90, 1841, 478)
    cv2.imwrite('result/cutout1.jpg', cutoutimg)
    cutoutimg= cv2.imread('result/cutout1.jpg')
    # 30, 487, 61, 524, 45, 505
    # 105, 321, 163, 502, 134, 411
    # 869, 425, 909, 514, 889, 469
    # 1816,402,1867,555,1841,478

    # y1 = (200-122)*1.2
    # y2 = (200+122)*0.8
    # x1 = (200-107)*1.4
    # x2 = (200+107)*0.8
    # y1 = (250-122)
    # y2 = (250+122)
    # x1 = (250-107)
    # x2 = (250+107)
    a = 1867 - 1816
    b = 555 - 402
    a = a/2
    b = b/2

    y1 = (250-b)
    y2 = (250+b)
    x1 = (250-a)
    x2 = (250+a)
    # y1 = (225-91)
    # y2 = (225+91)
    # x1 = (225-203)
    # x2 = (225+203)
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    cut_cutoutimg = cutoutimg[y1:y2,x1:x2]
    cv2.imwrite('result/cutout2.jpg', cut_cutoutimg)
    # cutoutimg=cutOutPannoama(im, 35, 171, 727)
    # cv2.imwrite('result/cutout2.jpg', cutoutimg)