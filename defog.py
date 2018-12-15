# -*- coding: utf-8 -*-
import cv2
import math
import numpy as np
import os


def rgbtohsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv2.split(rgb_lwpImg)
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    count=0.0
    for i in range(rows):
        for j in range(cols):
            sum = b[i, j]+g[i, j]+r[i, j]
            I = sum/3.0
            if I*255 >= 100:
                count = count+1
    return count/(rows*cols)


def zmMinFilterGray(src, r=7):
    if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    res = np.minimum(I  , I[[0]+list(range(h-1)), :])
    res = np.minimum(res, I[list(range(1,h))+[h-1], :])
    I = res
    res = np.minimum(I  , I[:, [0]+list(range(w-1))])
    res = np.minimum(res, I[:, list(range(1,w))+[w-1]])
    return zmMinFilterGray(res, r-1)


def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):
    V1 = np.min(m, 2)  # get dark channel
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)  # use guided filter
    bins = 2000
    ht = np.histogram(V1, bins)  # compute Airlight A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)  # constrain the value

    return V1, A


def deHaze(m, r=81, eps=0.001, w=1, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # get mask and airlight
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # color correction
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma correction
    return Y


if __name__ == '__main__':
    path1 = r"D:\\03fa746f0.jpg"
    # count_tar = 0
    # count_img=0
    # for fpathe, dirs, fs in os.walk(path1):
    #     for f in fs:
    src = cv2.imread(path1)
    img_dehaze = deHaze(src / 255.0) * 255
    img_dehaze = img_dehaze.astype(np.uint8)
    blur = cv2.medianBlur(img_dehaze, 21)  # 9,21
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 0, 30, apertureSize=3)  # 3
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=1, minLineLength=10,
                            maxLineGap=5)  # 25,11,(15,10)
    if lines is not None:
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(img_dehaze, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite(r'E:\\new2\\' + r"03fa746f0.jpg", img_dehaze)
    # print(count_tar / count_img)