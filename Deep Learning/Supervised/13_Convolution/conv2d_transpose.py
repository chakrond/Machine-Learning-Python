# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:06:29 2022

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

#%% Transpose convolution

# params
inChans  = 3 # RGB
imsize   = [64, 64]
outChans = 10
krnSize  = 11 # odd number
stride   = (2, 2)
padding  = 1

# create the instance
conv = nn.ConvTranspose2d(inChans, outChans, krnSize, stride, padding)

# print instance
print(conv)
print(' ')

# check weight tensor size
print(f'Size of weights: {conv.weight.shape}')
print(f'Size of bias: {conv.bias.shape}')

# plot kernel on the first color channels from 3
fig, ax = plt.subplots(2, 5, figsize=(10, 5))

for (i, axi) in enumerate(ax.flatten()):
    
    axi.imshow(conv.weight[0, i, :, :].detach(), cmap='gray')
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
fig, ax = plt.subplots(2, 5, figsize=(10, 5))

for (i, axi) in enumerate(ax.flatten()):
    
    axi.imshow(convImg[0, i, :, :].detach(), cmap='gray')
    axi.set_title(f'C1-{i}')
    axi.axis('off')    