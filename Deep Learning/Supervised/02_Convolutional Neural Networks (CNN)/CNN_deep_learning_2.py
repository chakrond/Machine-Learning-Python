# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 15:39:39 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import math
import time
import random
import os

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=2)

#%% Images preprocess, data generate

from keras.preprocessing.image import ImageDataGenerator

datagen_training_set = ImageDataGenerator(
        rescale=1./225,
        # rotation_range=40,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        # fill_mode='nearest'
        )

datagen_test_set = ImageDataGenerator(
        rescale=1./225,
        )

# Data set

target_size_set = [64, 64, 3]
batch_size_param = 16

# Traning set
traning_set = datagen_training_set.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=batch_size_param,
    class_mode='binary'
    )

# Test set
test_set = datagen_test_set.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=batch_size_param,
    class_mode='binary'
    )
    
#%% CNN Model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Sequential Model instance
cnn_model = Sequential()

# ------------------------------------------------------------------------ 
# No.1 Convolution Layer 
# ------------------------------------------------------------------------
cnn_model.add(
    Conv2D(
        filters=32, # how many feature detectors
        kernel_size=3, # feature detector size
        activation='relu', # activation function method
        input_shape=[64, 64, 3] # only for the first layer, connect first layer to the input layer
        # match with the target size in the model target_size=(64, 64) 
        # color image R,G,B = [255, 255, 255] dimension = 3 
        )
    )
# Max Pooling 
cnn_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        # strides=(2, 2)
        )
    )

# ------------------------------------------------------------------------ 
# No.2 Convolution Layer 
# ------------------------------------------------------------------------
cnn_model.add(
    Conv2D(
        filters=32, 
        kernel_size=3, 
        activation='relu'
        )
    )
# Max Pooling 
cnn_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        # strides=(2, 2)
        )
    )

# ------------------------------------------------------------------------ 
# No.3 Convolution Layer 
# ------------------------------------------------------------------------
cnn_model.add(
    Conv2D(
        filters=64, 
        kernel_size=3, 
        activation='relu'
        )
    )
# Max Pooling 
cnn_model.add(
    MaxPooling2D(
        pool_size=(2, 2),
        # strides=(2, 2)
        )
    )

# ------------------------------------------------------------------------ 
# Flatten 
# ------------------------------------------------------------------------
cnn_model.add(Flatten()) # converts our 3D feature maps to 1D feature vectors

# ------------------------------------------------------------------------
# Full connection layer 
# ------------------------------------------------------------------------
cnn_model.add(
    Dense(
        units=64, # units = number of neurons
        activation='relu'
        )
    ) 

# ------------------------------------------------------------------------
# Dropout layer
# ------------------------------------------------------------------------
cnn_model.add(
    Dropout(rate=0.5)
    )

# ------------------------------------------------------------------------
# Output layer
# ------------------------------------------------------------------------
cnn_model.add(
    Dense(
        units=1, # binary output
        activation='sigmoid'
        )
    ) 

# ------------------------------------------------------------------------
# Compile
# ------------------------------------------------------------------------
cnn_model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

# ------------------------------------------------------------------------
# Training the model
# ------------------------------------------------------------------------
cnn_model.fit( # batch learning
    traning_set, 
    # batch_size=32, 
    epochs=25,
    # steps_per_epoch=2000, # batch_size
    validation_data=test_set,
    # validation_steps=800 # batch_size
    ) 

# save weights of the model
cnn_model.save_weights('second_try.h5')  # save weights after training or during training

#%% Prediciton

# Image input & preprocessing
single_image = tf.keras.preprocessing.image.load_img('dataset/single_prediction/test_2.jpg', target_size=(64,64))
input_arr = tf.keras.preprocessing.image.img_to_array(single_image)
# input_arr = np.array([input_arr])  # Convert single image to a batch.
input_arr = np.expand_dims(input_arr, axis=0) # Convert single image to a batch.

# make prediciton
predictions = cnn_model.predict(input_arr)
classes = traning_set.class_indices
result  = []

# translate the output
if predictions[0][0] == classes['cats']:
    result = 'cat'
    
else:
    result = 'dog'
    
print('The prediction is', result)