#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _Author_: xiaofengShi
# Date: 2018-03-20 10:01:21
# Last Modified by:   xiaofengShi
# Last Modified time: 2018-03-20 10:01:21
'''
本程序识别图像中的公式，并将公式变为对应的数学表达式
'''
import cv2
import numpy as np
import skimage.io as io
import numpy as np
import argparse
import os
color = (255, 255, 255)


def sort_contours_function(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def Extrac_contours(image, cnt, image_extract_path):
    (sorted_controus, Bounding_box) = sort_contours_function(cnt)
    i = 0
    while i < len(sorted_controus):
        r = cv2.boundingRect(sorted_controus[i])
        mask = np.zeros(image.shape, np.uint8)
        mask = 0*image.copy()
        extract = 0*image.copy()
        cv2.drawContours(mask, sorted_controus, i,
                         (255, 255, 255), cv2.FILLED)
        for j in range(i+1, len(sorted_controus)):
            r2 = cv2.boundingRect(sorted_controus[j])
            if abs(r2[0]-r[0]) < 20:
                print('have one')
                cv2.drawContours(mask, sorted_controus, j,
                                 (255, 255, 255), cv2.FILLED)
                minX = int(min(r[0], r2[0]))
                minY = int(min(r[1], r2[1]))
                maxX = int(max(r[0]+r[2], r2[0]+r2[2]))
                maxY = int(max(r[1]+r[3], r2[1]+r2[3]))
                r = (minX, minY, maxX-minX, maxY-minY)
                i += 1
            else:
                break
        i += 1
        #############
        (rows, cols) = np.where(mask != 0)
        extract[rows, cols] = image[rows, cols]
        resize_pic = extract[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        name = '%s.jpg' % (i-1)
        saved_path = os.path.join(image_extract_path, name)
        cv2.imwrite(saved_path, resize_pic)


def Equation_recognize(image_path, image_extract_path):
    if not os.path.exists(image_extract_path):
        os.makedirs(image_extract_path)
    # step1: gauss filter and two values
    img = cv2.imread(image_path, 0)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    weight, height = img.shape[:]
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    img = cv2.bitwise_not(img)
    points = np.column_stack(np.where(img > 0))
    # get the whole center and the image's width and it's rotation angel
    rec = cv2.minAreaRect(points)
    angle = rec[-1]
    # int and reverse
    box_center = tuple(np.int0(rec[0]))
    box_center = tuple(reversed(list(box_center)))
    box_size = tuple(np.int0(rec[1]))
    box_size = tuple(reversed(list(box_size)))
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:]
    center = (w // 2, h // 2)
    # 首先反转图像，使得图像对齐
    transform_rotate = cv2.getRotationMatrix2D(
        center=center, angle=angle, scale=1.0)
    rotated = cv2.warpAffine(img, transform_rotate, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    # 剪切图像，这里要对box_size 和box_center进行反转
    croped = cv2.getRectSubPix(rotated, box_size, box_center)
    croped2 = croped.copy()
    croped2 = cv2.cvtColor(croped2, cv2.COLOR_GRAY2BGR)
    black = 0*croped.copy()
    croped3 = croped.copy()
    croped2 = cv2.cvtColor(croped3, cv2.COLOR_GRAY2BGR)
    # 找到联通区域
    _, contors, _ = cv2.findContours(
        croped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contors_ply = [cv2.approxPolyDP(i, 3, True) for i in contors]
    # 对contors进行筛选
    valid_contors = []
    for i in range(len(contors_ply)):
        r = cv2.boundingRect(contors_ply[i])
        area = r[2]*r[3]
        if area < 100:
            continue
        inside = False
        for j in range(len(contors_ply)):
            if j == i:
                continue
            r2 = cv2.boundingRect(contors_ply[j])
            area2 = r2[2]*r2[3]
            if area2 < 100 or area2 < area:
                continue
            if r[0] > r2[0] and r[0]+r[2] < r2[0]+r2[2] and r[1] > r2[1] and r[1]+r[3] < r2[1]+r2[3]:
                inside = True
        if inside:
            continue
        valid_contors.append(contors_ply[i])
    Bounding_box = [cv2.boundingRect(i) for i in valid_contors]
    # 对valid_contors 进行检验

    for i in range(len(valid_contors)):
        if Bounding_box[i][2]*Bounding_box[i][3] < 100:
            continue
        cv2.drawContours(black, valid_contors, i, (255, 255, 255), cv2.FILLED)
        cv2.rectangle(
            croped2, (Bounding_box[i][0],
                      Bounding_box[i][1]),
            (Bounding_box[i][0] + Bounding_box[i][2],
             Bounding_box[i][1] + Bounding_box[i][3]),
            color, 2, 8, 0)
    cv2.imshow('boudiongbox', black)

    # 对排序之后的边框countour进行检验

    (sorted_controus, Bounding_box) = sort_contours_function(valid_contors)
    for (i, c)in enumerate(sorted_controus):
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # draw the countour number on the image
        cv2.putText(croped3, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2)
        cv2.drawContours(black, sorted_controus, i, color, 1, 8)
    cv2.imshow("Sorted", croped3)
    # cv2.waitKey(0)

    Extrac_contours(croped3, valid_contors, image_extract_path)
    cv2.waitKey(0)


image_path = '1.png'
image_extract_path = './image_extract/'
Equation_recognize(image_path, image_extract_path)
