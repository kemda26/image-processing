import numpy as np
import cv2
import imutils
from imutils import contours

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 4

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue

def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh

# def possibleChar(contoure):


img = cv2.imread('image_cropped/50.jpg')
img_copy = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlurredRes, imgThreshRes = preprocess(img)
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
thresh_inv = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
edges = auto_canny(thresh)
cv2.imshow("thresh next level", thresh)
cv2.imshow("thresh inv", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()



cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []

for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    roi_area = w*h
    aspect_ratio = w/h
    print(roi_area,aspect_ratio,w,h)
    # if the contour is sufficiently large, it must be a digit
    if roi_area > 4000 and aspect_ratio > 0.3 and aspect_ratio < 2.0:
        crop = gray[y:y+h, x:x+w]
        cv2.imshow("cropped", crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        digitCnts.append(c)

digitCnts = contours.sort_contours(digitCnts,
    method="left-to-right")[0]

for c in digitCnts:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(img_copy,(x,y),( x + w, y + h ),(255, 0, 0),2)

cv2.imshow("finally", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# edges = auto_canny(thresh_inv)
# ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
# img_area = img.shape[0]*img.shape[1]
# for i, ctr in enumerate(sorted_ctrs):
#     x, y, w, h = cv2.boundingRect(ctr)
#     crop = gray[y:y+h, x:x+w]
#     # cv2.imshow("cropped", crop)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     roi_area = w*h
#     aspect_ratio = w/h
#     roi_ratio = roi_area/img_area
#     # print(roi_ratio)
#     if((roi_ratio >= 0.009) and (roi_ratio < 1.0) and (aspect_ratio > 0.25) and (aspect_ratio < 0.9)):
#         # cv2.imshow("cropped", crop)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         print(aspect_ratio, roi_ratio)
#         if ((h>1.2*w) and (3*w>=h)):
#             cv2.rectangle(img_copy,(x,y),( x + w, y + h ),(255, 0, 0),2)

# cv2.imshow("finally", img_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()