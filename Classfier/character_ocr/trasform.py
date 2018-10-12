# -*- coding: utf-8 -*-
# _Author_: xiaofengShi
# Date: 2018-03-18 09:38:49
# Last Modified by:   xiaofengShi
# Last Modified time: 2018-03-18 09:38:49
'''
本程序对图片进行映射变化，还有bug
'''

import argparse
import numpy as np
import cv2
import os
import math
from data_find_all_dirs import GetFileFromThisRootDir


def IsBadline(a, b):
    if (a**2 + b**2) < 100:
        return True
    else:
        return False


def CrossPoint(line1, line2):
    x0, y0, x1, y1 = line1[0]
    x2, y2, x3, y3 = line2[0]
    dx1 = x1 - x0
    dy1 = y1 - y0
    dx2 = x3 - x2
    dy2 = y3 - y2
    D1 = x1 * y0 - x0 * y1
    D2 = x3 * y2 - x2 * y3
    y = float(dy1 * D2 - D1 * dy2) / (dy1 * dx2 - dx1 * dy2)
    x = float(y * dx1 - D1) / dy1
    return [int(x), int(y)]


def SortPoints(want, center):
    top, bottom, want_sorted = [], [], []
    backup = want
    print(center)
    for i in range(len(want)):
        if want[i][1] < center[1] and len(top) < 2:
            top.append(want[i])
        else:
            bottom.append(want[i])
    if len(top) == 2 and len(bottom) == 2:
        tl, tr = top[0], top[1]
        bl, br = bottom[0], bottom[1]
        want_sorted = [tl, tr, br, bl]
    else:
        want_sorted = backup
    return want_sorted


def transform(img_path, rename_path):
    src = cv2.imread(img_path)
    out = src.copy()
    bkp = src.copy()
    height, width, channel = np.shape(src)
    img = src.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化成灰度图
    img = cv2.GaussianBlur(img, (5, 5), 0, 0)  # 高斯滤波
    # ret, thred = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY) # 二值化
    cv2.imwrite('./trannform/guss.png', img)
    # 进行膨胀，使用矩形的卷积核
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.dilate(img, element)
    cv2.imwrite('./trannform/dilate.png', img)
    # 使用canny进行边缘检测
    img = cv2.Canny(img, 20, 80)
    cv2.imwrite('./trannform/black.png', img)

    # 找到轮廓区域，只检索外边框
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)

    # 找到最大面积的轮廓坐标
    max_area = cv2.contourArea(contours[0])
    index = 0
    center = [0, 0]
    for count in range(len(contours)):
        tmp = cv2.contourArea(contours[count])
        if tmp > max_area:
            index = count
            max_area = tmp
    # 面积最大区域
    conter = [contours[index]]
    black = 0 * img.copy()
    for line_width in range(1, 4):
        print('line_width', line_width)
        cv2.drawContours(black, conter, 0, (255, 255, 0), line_width)
        cv2.imwrite('./trannform/contours.png', black)
        para_min = 10
        para_max = 300
        para_step = 1
        for para in range(para_min, para_max, para_step):
            '''
            cv2.HoughLinesP参数
            image： 必须是二值图像，推荐使用canny边缘检测的结果图像； 
            rho: 线段以像素为单位的距离精度，double类型的，推荐用1.0 
            theta： 线段以弧度为单位的角度精度，推荐用numpy.pi/180 
            threshod: 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少。根据情况推荐先用100试试
            lines：这个参数的意义未知，发现不同的lines对结果没影响，但是不要忽略了它的存在 
            minLineLength：线段以像素为单位的最小长度，根据应用场景设置 
            maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
            '''
            lines = cv2.HoughLinesP(black, 1, np.pi / 180, threshold=para,
                                    minLineLength=math.ceil(
                                        0.3*(min(height, width))),
                                    maxLineGap=math.ceil(0.1*(min(height, width))))
            want = []
            flag = 0
            if len(lines) >= 4:
                EraseLines = set()
                # 判断得到的直线是否距离过近
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        if (IsBadline(
                                abs(lines[i][0][0] - lines[j][0][0]),
                                abs(lines[i][0][1] - lines[j][0][1]))
                                and IsBadline(
                                    abs(lines[i][0][2] - lines[j][0][2]),
                                    abs(lines[i][0][3] - lines[j][0][3]))):
                            EraseLines.add(j)
                if EraseLines:
                    delcount = 0
                    for j in EraseLines:
                        lines = np.delete(lines, j - delcount, axis=0)
                        delcount += 1
                if len(lines) != 4:
                    continue
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        points = CrossPoint(lines[i], lines[j])
                        if 0 < points[0] < width and 0 < points[1] < height:
                            want.append(points)
                print('want_1', want)
                if len(want) != 4:
                    continue
                IsGoodPoints = True
                for i in range(len(want)):
                    for j in range(i + 1, len(want)):
                        distance = np.sqrt((want[i][0] - want[j][0])**2 +
                                           (want[i][1] - want[j][1])**2)
                        if distance < 5:
                            IsGoodPoints = False
                if not IsGoodPoints:
                    continue
                print('want_oo', want)
                want.sort()
                want = np.array(want, dtype='float32')
                approx = cv2.approxPolyDP(
                    want, cv2.arcLength(want, True)*0.02, True)
                if len(lines) == 4 and len(want) == 4 and len(approx) == 4:
                    flag = 1
                    break
        if len(lines) == 4 and len(want) == 4 and len(approx) == 4:
            break

    # get the center points
    print('want--', want)
    for i in range(len(want)):
        center += want[i]
    center *= (1/len(want))
    '''
    # see the find points
    center = tuple([math.ceil(i) for i in center])
    cv2.circle(bkp, tuple(want[0]), 3, (255, 0, 0), -1)
    cv2.circle(bkp, tuple(want[1]), 3, (0, 255, 0), -1)
    cv2.circle(bkp, tuple(want[2]), 3, (0, 0, 255), -1)
    cv2.circle(bkp, tuple(want[3]), 3, (255, 255, 255), -1)
    cv2.circle(bkp, center, 3, (255, 0, 255), -1)
    cv2.imshow('bkp', bkp)
    # cv2.waitKey(0)
    '''
    if flag:
        want_sorted = SortPoints(want, center)
        width_transform = int(
            np.sqrt(((want_sorted[0][0] - want_sorted[1][0])**2) + (want_sorted[0][1] - want_sorted[1][1])**2))
        height_transform = int(
            np.sqrt(((want_sorted[0][0] - want_sorted[3][0])**2) + (want_sorted[0][1] - want_sorted[3][1])**2))

        dstrect = np.array(
            [[0, 0], [width_transform - 1, 0],
                [width_transform + 1, height_transform + 1], [0, height_transform - 1]],
            dtype='float32')
        transform_persp = cv2.getPerspectiveTransform(
            np.array(want_sorted), dstrect)
        warpedimg = cv2.warpPerspective(
            out, transform_persp, (width_transform, height_transform))
        cv2.imwrite(rename_path, warpedimg)


'''
img_path = './test_img/12.png'
rename_path = './test_img/12_trans.png'
transform(img_path,rename_path)
'''

path = './test_img/'
save_path = './trannform/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
extension = ['jpg', 'png', ]
file_list = GetFileFromThisRootDir([path], extension)
for file in file_list:
    file_noext = file.split('/')[-1].split('.')[0]
    extension = file.split('.')[-1]
    # file_noext=file.split('/')[-1]
    new_name = file_noext+'_trsansform.'+extension
    new_path = os.path.join(save_path, new_name)
    print('current file is :', file)
    transform(file, new_path)
