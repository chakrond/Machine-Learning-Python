# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:24:25 2022

@author: Chakron.D
"""

# %% Importing the libraries

# python
import math
import time
import random
import os
import sys
import copy

# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

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
from sklearn.decomposition import PCA, KernelPCA
import sklearn.metrics as skm

# scipy
import scipy.stats as stats
from scipy.signal import convolve2d

# read image
from imageio import imread

# set directory
import inspect, os.path
filename = inspect.getframeinfo(inspect.currentframe()).filename
path_dir = os.path.dirname(os.path.abspath(filename))
os.chdir(path_dir) # set the working directory
cd = os.getcwd()

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5  # default 0.8
plt.rcParams["figure.subplot.hspace"] = 0.25

#%% Conv2d class instance

# params
in_chans = 3 # RGB
out_chans = 15
kernel_size = 5 # odd number
stride = 1
padding = 0

# instance
conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding)

# print
print(conv)
print(' ')

# check the dimensions
print(f'Size of weights: {conv.weight.shape}') # [15, 3, 5, 5] = [output_channels, input_channels, kernel_size]
print(f'Size of bias: {conv.bias.shape}') # number of bias is eqaul to number of output kernel (15)

# plot kernel on the first color channels from 3
fig, ax = plt.subplots(3, 5, figsize=(10, 5))

for (i, axi) in enumerate(ax.flatten()):
    
    axi.imshow(conv.weight[i, 0, :, :].detach(), cmap='gray')
    axi.set_title(f'C1-{i}')
    axi.axis('off')
    
#%% convolve results Pytroch and Mathplotlib

# Pytroch**
# image size (N-batch, RGB, width, height)
imsize = (1, 3, 64, 64) # tuple
img = torch.rand(imsize)

# Mathplotlib**
img_plt = img.permute(2, 3, 1, 0).numpy() # to (width, height, RGB, N-batch)
print(f'Pytoch: {img.shape}')
print(f'Matplotlib {img_plt.shape}')

plt.imshow(np.squeeze(img_plt))

#%% convolve image

convImg = conv(img)
print(f'Input Image: {img.shape}')
print(f'Result Image {convImg.shape}')

# plot
fig, ax = plt.subplots(3, 5, figsize=(10, 5))

for (i, axi) in enumerate(ax.flatten()):
    
    axi.imshow(convImg[0, i, :, :].detach(), cmap='gray') 
    axi.set_title(f'C1-{i}')
    axi.axis('off')