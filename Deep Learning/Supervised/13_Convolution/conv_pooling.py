# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 22:09:25 2022

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

#%% create a pool class instance

# parameters
poolSize = 3
stride   = 3

# create the instance
p2d = nn.MaxPool2d(poolSize, stride=3)
p3d = nn.MaxPool3d(poolSize, stride=3)

# print
print(p2d)
print(p3d)

#%% Create image

# create a 2D and a 3D image
img2d = torch.randn(1, 1, 30, 30)
img3d = torch.randn(1, 3, 30, 30)

# apply maxpooling
img2Pool2 = p2d(img2d)
print(f'2D image, 2D maxpool: {img2Pool2.shape}\n')

img3Pool3 = p3d(img3d)
print(f'3D image, 3D maxpool: {img3Pool3.shape}\n')

# plot only on one channel
fig, ax = plt.subplots(2, 2, figsize=(10, 5))

# 2D
ax[0, 0].imshow(img2d[0, 0, :, :], cmap='gray')
ax[0, 0].set_title('Img 2d')
ax[0, 1].imshow(img2Pool2[0, 0, :, :], cmap='gray')
ax[0, 1].set_title('Pool 2d')

# 3D
ax[1, 0].imshow(img3d[0, 0, :, :], cmap='gray')
ax[1, 0].set_title('Img 3d')
ax[1, 1].imshow(img3Pool3[0, 0, :, :], cmap='gray')
ax[1, 1].set_title('Pool 3d')

