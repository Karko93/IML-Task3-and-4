# -*- coding: utf-8 -*-
"""

Task 4

Created on Thu May 21 17:08:20 2020

@author: Alex
"""

import os 
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


# load images
# returns array of float32 (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3)
def load_and_crop(BATCH_SIZE = 10000, IMG_HEIGHT = 242, IMG_WIDTH = 354):
    
    

    train_image = []
    
    for i in range(0,BATCH_SIZE):
        
        # load image
        img = load_img(os.path.join('food/food','{0:05}'.format(i)+'.jpg'))
        
        # Size of the image in pixels (size of orginal image) 
        width, height = img.size 
          
        # Setting the points for cropped image 
        x1 = (width - IMG_WIDTH)/2
        y1 = (height - IMG_HEIGHT)/2
        x2 = IMG_WIDTH + x1
        y2 = IMG_HEIGHT + y1
          
        # Cropped image of above dimension 
        img = img.crop((x1, y1, x2, y2))
            
        # convert PIL to numpy and normalize
        img = img_to_array(img)
        img = img/255
        
        train_image.append(img)
        
    return np.array(train_image)



# load and resize images
# returns array of float32 (BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3)
def load_and_resize(BATCH_SIZE = 10000, IMG_HEIGHT = 242, IMG_WIDTH = 354):
    
    data_dir = pathlib.Path('food')
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])
    
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # image resized using the default interpolation : "nearest"
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=False,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes = list(CLASS_NAMES))
    
    image_batch, label_batch = next(train_data_gen)
    
    return image_batch
    


# load training data and rearrange it
# returns a numpy array (119030, 3)
def read_and_prep_train_data():
    
    train_triplets = np.loadtxt("train_triplets.txt", delimiter=" ", dtype=np.int64)

    data = []
    
    for row in train_triplets: 
        data.append([row[0], row[1], row[2], 1])
        data.append([row[0], row[2], row[1], 0])
        
    return np.asarray(data, dtype=np.int64)