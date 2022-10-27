# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 19:53:21 2022

@author: Chakron.D
"""

# %% Importing the libraries

# python
import math
import time
import random
import os

# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

# panda
import pandas as pd

# Symbol python
import sympy as sym

# PyTroch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Seaborn
import seaborn as sns

# sckitlearn
from sklearn.model_selection import train_test_split

# scipy
import scipy.stats as stats

# set directory
import inspect, os.path
filename = inspect.getframeinfo(inspect.currentframe()).filename
path_dir     = os.path.dirname(os.path.abspath(filename))
os.chdir(path_dir) # set the working directory
cd = os.getcwd()

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5  # default 0.8

#%% Functions

# Data split & data loader function
def dSplit(data_ts,
           y_ts,
           model_test_size=0.2,
           p_batch_size=32,
           ):
    
    # split to find array length
    data_train, data_test, y_train, y_test = train_test_split(
        data_ts, y_ts, test_size=model_test_size)

    # ---------------------------------------
    # Create Datasets, Dataloader, Batch
    # ---------------------------------------
    # convert to PyTorch Datasets
    Dataset_train = TensorDataset(data_train, y_train)
    Dataset_test = TensorDataset(data_test, y_test)

    # finally, translate into dataloader objects
    # drop_last=True, drop the last batch if it's not full batch
    DataLoader_train = DataLoader(
        Dataset_train, shuffle=True, batch_size=p_batch_size, drop_last=True)
    DataLoader_test = DataLoader(
        Dataset_test, batch_size=Dataset_test.tensors[0].shape[0])
    
    return DataLoader_train, DataLoader_test

#%% Import dataset

# MNIST data
fname = 'z:\python traning\my codes\machine learning\deep learning\supervised\99_Datasets\MNIST\mnist_train_small.csv'
fdata = np.loadtxt(open(fname, 'rb'), delimiter=',')

# extract data and labels
labels = fdata[: ,0]
data = fdata[:, 1:]

#%% Data inspection

# show a few random  of 28x28 img
fig, axs = plt.subplots(3, 4, figsize=(10, 6))

# generate random number
randimg2show = np.random.randint(data.shape[0], size=len(axs.flatten()))


for i, ax in enumerate(axs.flatten()):
  
  # create the image (must be reshaped!)
  img = np.reshape(data[randimg2show[i],:], (28, 28))
  ax.imshow(img, cmap='gray')

  # title
  ax.set_title('The number %i'%labels[randimg2show[i]])

plt.suptitle('28x28 Image Data', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, .95])
plt.show()


# show a few random digits of data vector
fig,axs = plt.subplots(3, 4, figsize=(10, 6))

for i, ax in enumerate(axs.flatten()):

  # create the image
  ax.plot(data[randimg2show[i],:],'ko')

  # title
  ax.set_title('The number %i'%labels[randimg2show[i]])

plt.suptitle('How the FFN model sees the data(vector)', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, .95])
plt.show()

#%% Correlation

# let's see some example 7s

# find indices of all the 7's in the dataset
idx7 = np.where(labels==7)[0]

# draw the first 12
fig,axs = plt.subplots(2, 6, figsize=(15, 6))

for i,ax in enumerate(axs.flatten()):
  img = np.reshape(data[idx7[i],:],(28, 28))
  ax.imshow(img,cmap='gray')
  ax.axis('off')

plt.suptitle("Example 7's",fontsize=20)
plt.tight_layout(rect=[0,0,1,.95])

# let's see how they relate to each other by computing spatial correlations
C = np.corrcoef(data[idx7,:])

# and visualize
fig,ax = plt.subplots(1, 3, figsize=(16, 6))
ax[0].imshow(C, vmin=0, vmax=1)
ax[0].set_title("Correlation across all 7's")

# extract the unique correlations and show as a scatterplot
uniqueCs = np.triu(C, k=1)
uniqueCsF = uniqueCs.flatten()
ax[1].hist(uniqueCsF[uniqueCsF!=0], bins=100)
ax[1].set_title('All unique correlations')
ax[1].set_xlabel("Correlations of 7's")
ax[1].set_ylabel('Count')

# show all 7's together
aveAll7s = np.reshape( np.mean(data[idx7, :], axis=0), (28,28))
ax[2].imshow(img, cmap='gray')
ax[2].set_title("All 7's averaged together")

plt.tight_layout()
plt.show()


#%% randomly scramble the data,

# preserving the re-ordering for each image
dataNorm = data / np.max(data)
randIdx = np.random.permutation(data.shape[1])
scrambled = dataNorm[:, randIdx]


# show a few random digits
fig,axs = plt.subplots(3, 4, figsize=(10, 6))

for ax in axs.flatten():
  # pick a random image
  randimg2show = np.random.randint(0, high=data.shape[0])

  # create the image (must be reshaped!)
  img = np.reshape(scrambled[randimg2show, :], (28, 28))
  ax.imshow(img, cmap='gray')

  # title
  ax.set_title('The number %i'%labels[randimg2show])

plt.suptitle('The scrambled data', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% convert to tensor

data_ts = torch.tensor(dataNorm).float()
y_ts = torch.tensor(labels).long()
# y_ts = y_ts.reshape((-1, 1))

# %% train/test dataset

# params
test_size = 0.1
b_size = 32

DataLoader_train, DataLoader_test = dSplit(data_ts, y_ts, model_test_size=test_size, p_batch_size=b_size)

#%% Shift the test images by a few pixels

# shift a vectorized image

# grab one image data
img = DataLoader_test.dataset.tensors[0][0, :]
img_origin = img.reshape(28, 28) # reshape to 2D image

# shift the image (pytorch calls it "rolling")
tmp_roll_0 = torch.roll(img_origin, 14, dims=0)
tmp_roll_1 = torch.roll(img_origin, 14, dims=1)

# now show them both
fig,ax = plt.subplots(1, 3, figsize=(10, 6))
ax[0].imshow(img_origin, cmap='gray')
ax[0].set_title('Original')

ax[1].imshow(tmp_roll_0, cmap='gray')
ax[1].set_title('Shifted axis 0 (rolled)')

ax[2].imshow(tmp_roll_1, cmap='gray')
ax[2].set_title('Shifted axis 1 (rolled)')

plt.show()

#%% apply to all images in the test set

for i in range(DataLoader_test.dataset.tensors[0].shape[0]):
  
    # get the image
    img = DataLoader_test.dataset.tensors[0][i, :]
    
    # reshape and roll by max. 10 pixels
    randroll = np.random.randint(-10, 11)
    img = torch.roll(img.reshape(28, 28) ,randroll, dims=1)
    
    # re-vectorize and put back into the matrix
    DataLoader_test.dataset.tensors[0][i, :] = img.reshape(1, -1)
    
    