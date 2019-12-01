import os
import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from PIL import Image
from skimage import exposure
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['figure.dpi'] = 125
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
plt.rcParams['image.cmap'] = 'gray'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def adjust_contrast(img):
    if exposure.is_low_contrast(img):
        return exposure.adjust_gamma(img, 0.6)
    else:
        return exposure.adjust_gamma(img, 1.3)


def denoise(image):
    norm = image.copy()
#     norm = exposure.rescale_intensity(image)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
    blur = cv2.bilateralFilter(gray,20,45,45)
    edge = cv2.Canny(blur, 30, 40, apertureSize = 3)
    return blur, edge, norm


def bbox(thresh, img):
    extra = 10
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    img_height, img_width = img.shape[:2]

    test = img.copy()
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(test, (x - extra, y - extra), (x + w + extra, y + h + extra), (0, 255, 0), 1)
#         if x < img_width - (img_width / 4) and y < img_height / 1.8 and w < 220 and y > 20:
#         boxes.append((x,y,w,h))
        cv2.rectangle(img, (x - extra, y - extra), (x + w + extra, y + h + extra), (0, 255, 0), 1)

#     for box in bbox_filter(boxes)[:5]:
#         x,y,w,h,_ = box
#         print(x,y,w,h)
#         cv2.rectangle(img, (x - extra, y - extra), (x + w + extra, y + h + extra), (0, 255, 0), 1)

    return img    

def pre_morph(img):
    morph = img
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))) #8,8
    return morph


def post_morph(img):
    morph = img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
#     morph = cv2.erode(morph, kernel_2, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    morph = cv2.dilate(morph, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)), iterations=1)

    return morph


def preprocess(inputs):
    img = inputs.copy()
    img = adjust_contrast(img)
    img = pre_morph(img)
    blur, edge, norm = denoise(img)
#     blur, norm, _input = rotate(blur, edge, norm, inputs)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 69, -3) #69
    morph = post_morph(thresh)
    return blur, edge, norm, thresh, morph


def bbox_color_cord(norm, inputs):
    def sortarea(i):
        return i[4]

    inputs = inputs.copy()
    norm = norm.copy()
    lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    na = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    retval, na_thresh = cv2.threshold(na, thresh = 150,  maxval=255, type=cv2.THRESH_BINARY)
    na_thresh = cv2.morphologyEx(na_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    na_thresh = cv2.dilate(na_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 23)), iterations=1)
#     plt.imshow(na_thresh)

    contours, hierarchy = cv2.findContours(na_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < h and w/h > 0.3:
            areas.append((x,y,w,h,w*h))
    areas.sort(key=sortarea)

    return areas[-1]


def bbox_filter(inputs, gt_x, gt_y, gt_w, gt_h):
    def sortloss(x):
        return x[-1]

    saved = []
    for x,y,w,h in inputs:
        wh_ratio = h / w if h > w else w / h
        loss = ((gt_w*gt_h - w*h)**2) + ((gt_y - y)**2)*10 #+ ((gt_x - x)**2)
        saved.append((x, y, w, h, loss))
    saved.sort(key=sortloss)

    result = []
    for box in saved[:10]:
        is_child_box = False
        for target in saved[:10]:
            if box == target: continue
            else:
                (x1,y1,w1,h1,l1), (x2,y2,w2,h2,l2) = box, target
                if x1 > x2  and y1 > y2  and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
                    is_child_box = True
                    break
        if not is_child_box: result.append(box)

    res = []
    for x,y,w,h,l in result:
        if x < gt_x and y < gt_y and x + w >= gt_x + gt_w:
            continue
        res.append((x,y,w,h,l))

    return res


def sortx(i):
    return i[0]


def bbox(thresh, norm, img):
    extra = 10
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    img_height, img_width = img.shape[:2]

    gt_x, gt_y, gt_w, gt_h, _ = bbox_color_cord(norm, img)
    # print('gt', gt_x, gt_y, gt_w, gt_h)
    test = img.copy()
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x < gt_x - 30 and y < gt_y + gt_h/3 and y > gt_y - gt_h/2 and w < h and h > gt_h/2 and w/h >= 0.2:
            boxes.append((x, y, w, h))

    # for box in bbox_filter(boxes, gt_x, gt_y, gt_w, gt_h)[:15]:
        # x,y,w,h,_ = box
        # print('8 box: ',box)
        # cv2.rectangle(test, (x - extra, y - extra), (x + w + extra, y + h + extra), (0, 255, 0), 1)
    # cv2.rectangle(test, (gt_x - extra, gt_y - extra), (gt_x + gt_w + extra, gt_y + gt_h + extra), (255, 0, 0), 1)

    proposed_boxes = bbox_filter(boxes, gt_x, gt_y, gt_w, gt_h)[:5]
    proposed_boxes.sort(key=sortx)
    # for box in proposed_boxes:
        # x,y,w,h,_ = box
        # print(box)
        # cv2.rectangle(img, (x - extra, y - extra), (x + w + extra, y + h + extra), (0, 255, 0), 1)

    return img, test, proposed_boxes