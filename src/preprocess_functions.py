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


def get_angle(lines):
    # h: horizontal, v: vertical
    filters   = {'h': [], 'v': []}
    theta_min = {'h': 63 * np.pi / 180, 'v': 27 * np.pi / 180} 
    theta_max = {'h': 117 * np.pi / 180, 'v': 153 * np.pi / 180}
    theta_avr = {'h': 0, 'v': 0}
    theta_deg = {'h': 0, 'v': 0}
    
    if lines is None:
        return 0
    for line in lines:
        _, theta = line[0]
        if theta > theta_min['h'] and theta < theta_max['h']:
            filters['h'].append(theta)
            theta_avr['h'] += theta
        if theta < theta_min['v'] and theta > theta_max['v']:
            filters['v'].append(theta)
            theta_avr['v'] += theta

            
    if len(filters['h']) > 0: 
        theta_avr['h'] /= len(filters['h'])
        theta_deg['h'] = (theta_avr['h'] / np.pi * 180) - 90
        
    if len(filters['v']) > 0:
        theta_avr['v'] /= len(filters['v'])
        theta_deg['v'] = (theta_avr['v'] / np.pi * 180)
    
    angle = (theta_deg['h'] + theta_deg['v']) / 2
    if theta_deg['h'] == 0 or theta_deg['v'] == 0:
            return theta_deg['h'] if theta_deg['v'] == 0 else theta_deg['v']
    
    return angle


def pre_morph(img):
    morph = img
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))) #8,8
    return morph


def post_morph(img):
    morph = img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    
    morph = cv2.erode(morph, kernel_2, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    morph = cv2.dilate(morph, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)), iterations=1)
#     morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    return morph
    

def show_houghlines(img, lines):
    if lines is None:
        return 0
    for line in lines:
        r, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta) 
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b)) 
        y1 = int(y0 + 1000*(a)) 
        x2 = int(x0 - 1000*(-b)) 
        y2 = int(y0 - 1000*(a)) 
        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
    plt.imshow(img)
    
    
def hough_lines(edge):
    lines = cv2.HoughLines(edge, 1, np.pi / 180, 117)
    return lines


def denoise(image):
    norm = exposure.rescale_intensity(image)
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
    blur = cv2.bilateralFilter(gray,10,45,45)
    edge = cv2.Canny(blur, 50, 150, apertureSize = 3)
    return blur, edge, norm


def rotate(blur, edge, norm, image):
    lines = hough_lines(edge)
    angle = get_angle(lines)
#     show_houghlines(image, lines)
    h, w = edge.shape[:2]
    center = (w / 2, h / 2)    
    transform_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    dest = cv2.warpAffine(blur, transform_matrix, (w, h))
    norm = cv2.warpAffine(norm, transform_matrix, (w, h))
    image = cv2.warpAffine(image, transform_matrix, (w, h))
    return dest, norm, image


def preprocess(inputs):
    image = inputs.copy()
    contrast = adjust_contrast(image) 
    morph = pre_morph(contrast)
    blur, edge, norm = denoise(morph)
    rotated, norm, _input = rotate(blur, edge, norm, inputs)
#     rotated = exposure.rescale_intensity(rotated)
    thresh = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, -1)
    morph_thresh = post_morph(thresh)
    box, box_cord = bbox(morph_thresh, norm)

    return _input, norm, box, box_cord


def bbox_filter(inputs):
    def sortloss(x):
        return x[-1]
    
    y_w = 150
    y_h = 230
    saved = []
    for x,y,w,h in inputs:
        wh_ratio = h / w if h > w else w / h
        loss = ((y_w - w)**2 + (y_h - h)**2) + abs(y_h/y_w - h/w)*5000
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
                
    return result

    
def bbox(thresh, inputs):
    img = inputs.copy()
    extra = 10
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    img_height, img_width = img.shape[:2]
    
    # test = img.copy()
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # cv2.rectangle(test, (x - extra, y - extra), (x + w + extra, y + h + extra), (0, 255, 0), 1)
        if x < img_width - (img_width / 4) and y < img_height / 1.8 and w < 220 and y > 25:
            boxes.append((x,y,w,h))
            
    # for box in bbox_filter(boxes)[:5]:
        # x,y,w,h,_ = box
        # cv2.rectangle(img, (x - extra, y - extra), (x + w + extra, y + h + extra), (0, 255, 0), 1)

    return img, bbox_filter(boxes)[:5]
    
    
def crop_image(img, box_cord):
    height, width = img.shape[:2]

    def sortx(i):
        return i[0]
    
    box_cord.sort(key=sortx)
    crop_images = []
    for x,y,w,h,_ in box_cord:
        extra = 10
        if y - extra < 0 or y + h + extra > height or x - extra < 0 or x + h + extra > width:
            extra = 0
        cropped = img[y - extra:y + h + extra, x - extra:x + w + extra]
        crop_images.append(cropped)

    return crop_images
