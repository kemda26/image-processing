import numpy as np
import scipy
import cv2
import tensorflow as tf
import keras
# import torch
# import torchvision
import sklearn
import skimage

import task1_preprocess as F1
import task2_preprocess as F2
from keras.models import load_model


class Reader:

    def __init__(self, data_folder):
        self.name = "Reader"
        self.data_folder = data_folder


    # Prepare your models
    def prepare(self):
        self.model = load_model('svhn.hdf5')


    # Implement the reading process here
    def process(self, img):
        image, pp_image, box, box_cord = F1.preprocess(img)
        cropped_images = F1.crop_image(pp_image, box_cord)

        crops = []
        for cropped in cropped_images:
            resized = skimage.transform.resize(cropped, (32,32,3), mode='reflect')
            crops.append(resized)
        crops = np.array(crops)

        prob = self.model.predict(crops)
        targets = np.argmax(prob, axis=1)
        # print(targets)

        return int(''.join(str(i) for i in targets))


    # Prepare your models
    def prepare_crop(self):
        self.model = load_model('svhn.hdf5')


    # Implement the reading process here
    def crop_and_process(self, image):
        test = image.copy()
        blur, edge, norm, thresh, morph = F2.preprocess(test)
        img, box, box_cord = F2.bbox(thresh, norm.copy(), test)
        if len(box_cord) == 0:
            return 0
        cropped_images = F1.crop_image(img, box_cord)

        crops = []
        for cropped in cropped_images:
            resized = skimage.transform.resize(cropped, (32,32,3), mode='reflect')
            crops.append(resized)
        crops = np.array(crops)
        prob = self.model.predict(crops)
        targets = np.argmax(prob, axis=1)
        
        return int(''.join(str(i) for i in targets))



def check_import():
    print("Python 3.6.7")
    print("Numpy = ", np.__version__)
    print("Scipy = ", scipy.__version__)
    print("Opencv = ", cv2.__version__)
    print("Tensorflow = ", tf.__version__)
    print("Keras = ", keras.__version__)
    # print("pytorch = ", torch.__version__)
    # print("Torch vision = ", torchvision.__version__)
    print("Scikit-learn = ", sklearn.__version__)
    print("Scikit-image = ", skimage.__version__)

if __name__=="__main__":
    check_import()

"""
Using TensorFlow backend.
Python 3.6.7
Numpy =  1.14.5
Scipy =  1.2.1
Opencv =  4.1.1
Tensorflow =  1.14.0
Keras =  2.3.0
pytorch =  1.0.1.post2
Torch vision =  0.2.2
Scikit-learn =  0.21.3
Scikit-image =  0.14.2
"""
"""
    TEAM:
    - Nguyen Duy Quang 18/08
    - Pham Tuan Dung
    - Phi Van Minh
    - Pham Xuan Thanh
"""
