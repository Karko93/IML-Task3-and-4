# -*- coding: utf-8 -*-
"""

Task 4

Created on Thu May 21 17:08:20 2020

@author: Alex
"""

import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# load images
# returns ndarray object (10000,) of numpy modules (each module is an "image")
def load_images():

    train_image = []
    
    for i in range(0,10000):
        img = load_img(os.path.join('food/food','{0:05}'.format(i)+'.jpg'))
        img = img_to_array(img)
        img = img/255           # normalize image array
        train_image.append(img)
        
    return np.array(train_image)


# load training data and rearrange it
# returns a numpy array (119030, 3)
def read_and_prep_train_data():
    
    train_triplets = np.loadtxt("train_triplets.txt", delimiter=" ", dtype=np.int64)

    data = []
    
    for row in train_triplets: 
        data.append([row[0], row[1], 1])
        data.append([row[0], row[2], 0])
        
    return np.asarray(data, dtype=np.int64)