import numpy as np
import cv2

'''
对图像进行衍射变换和平移旋转，将图像变换
'''


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

    return (int(x), int(y))


def SortPoint(points):
    sp = sorted(points, key=lambda x: (int(x[1]), int(x[0])))
    if sp[0][0] > sp[1][0]:
        sp[0], sp[1] = sp[1], sp[0]

    if sp[2][0] > sp[3][0]:
        sp[2], sp[3] = sp[3], sp[2]

    return sp


'''
将图像转化成灰度图，并进行高斯滤波，之后进行边缘检测

使用 Canny 边缘检测，必须满足以下两个条件：

能有效地抑制噪声
必须尽量精确确定边缘的位置
'''


def PerspectiveTransform(img_path, total_rotate_flage):
    src = cv2.imread(img_path)
    rgbsrc = src.copy()
    graysrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurimg = cv2.GaussianBlur(graysrc, (3, 3), 0)
    Cannyimg = cv2.Canny(blurimg, 35, 189)
    cv2.imshow('canny', Cannyimg)
    lines = cv2.HoughLinesP(Cannyimg, 1, np.pi / 180,
                            threshold=30, minLineLength=320, maxLineGap=40)
    print(np.shape(lines))
    print(type(lines))
    print(len(lines))
    # print(lines.size())
    for i in range(int(np.size(lines) / 4)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(rgbsrc, (x1, y1), (x2, y2), (255, 255, 0), 3)
    points = np.zeros((4, 2), dtype='float32')
    points[0] = CrossPoint(lines[0], lines[2])
    points[1] = CrossPoint(lines[0], lines[3])
    points[2] = CrossPoint(lines[1], lines[2])
    points[3] = CrossPoint(lines[1], lines[3])

    sp = SortPoint(points)
    print(np.shape(sp))
    width = int(
        np.sqrt(((sp[0][0] - sp[1][0]) ** 2) + (sp[0][1] - sp[1][1]) ** 2))
    height = int(
        np.sqrt(((sp[0][0] - sp[2][0]) ** 2) + (sp[0][1] - sp[2][1]) ** 2))

    dstrect = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]], dtype="float32")
    '''
    进行透视变换
    '''
    print(sp, dstrect)
    transform_persp = cv2.getPerspectiveTransform(np.array(sp), dstrect)
    warpedimg = cv2.warpPerspective(src, transform_persp, (width, height))
    '''
    进行水平旋转
    '''
    if total_rotate_flage == True:
        gray = cv2.cvtColor(warpedimg, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
        print(thresh)
        coord = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coord)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:]
        center = (w // 2, h // 2)
        transform_rotate = cv2.getRotationMatrix2D(
            center=center, angle=angle, scale=1.0)
        rotated = cv2.warpAffine(warpedimg, transform_rotate, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated
    else:
        return warpedimg


if __name__ == '__main__':
    dst = PerspectiveTransform(
        '/Users/xiaofeng/Code/Github/character_ocr/test_img/1.png', True)
    cv2.imwrite(
        '/Users/xiaofeng/Code/Github/character_ocr/test_img/input-mm.png', dst)
