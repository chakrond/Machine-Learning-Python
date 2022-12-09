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

#%% image
imgN = 20
image = np.random.randn(imgN,imgN)

# convolution kernel
kernelN = 7
Y,X = np.meshgrid(np.linspace(-3,3,kernelN),np.linspace(-3,3,kernelN))
kernel = np.exp( -(X**2+Y**2)/7 )


# let's see what they look like
fig,ax = plt.subplots(2,2,figsize=(8,6))
ax[0, 0].imshow(image)
ax[0, 0].set_title('Image')

ax[0, 1].imshow(kernel)
ax[0, 1].set_title('Convolution kernel')

#the convolution
convoutput = np.zeros((imgN,imgN))
halfKr = kernelN//2

# draw boundary line
bline = ax[0, 0].plot([0, kernelN, kernelN, 0, 0],
              [0, 0, kernelN, kernelN, 0], 'r')

for rowi in range(halfKr,imgN-halfKr):
  for coli in range(halfKr,imgN-halfKr):

    # cut out a piece of the image 7x7 fit the kernel
    pieceOfImg = image[rowi-halfKr:rowi+halfKr+1, coli-halfKr:coli+halfKr+1] 
    ax[1, 0].imshow(pieceOfImg)
    
    # move line
    bline[0].set_ydata([rowi-halfKr, rowi+halfKr+1, rowi+halfKr+1, rowi-halfKr, rowi-halfKr])
    bline[0].set_xdata([coli-halfKr, coli-halfKr, coli+halfKr+1, coli+halfKr+1, coli-halfKr])
    
    # dot product: element-wise multiply and sum (and flip the kernel for "real convolution")
    dotprod = np.sum( pieceOfImg*kernel[::-1, ::-1] )

    # store the result for this pixel
    convoutput[rowi, coli] = dotprod
    
    if coli == halfKr:
        im = ax[1, 1].imshow(convoutput)
        ax[1, 1].set_title('Manual convolution')
    
    im.set_data(convoutput)
    
    plt.show()
    time.sleep(0.001)
    plt.pause(0.001)

    
#%% using scipy
convoutput2 = convolve2d(image, kernel, mode='valid')

#%% wiht real image

# read a pic
ReadImg = imread('De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg')

# check the size
print(ReadImg.shape)

# show image
fig = plt.figure(figsize=(10, 6))
plt.imshow(ReadImg);

# transform image to 2D
imgT = np.mean(ReadImg, axis=2)
imgT = imgT/np.max(ReadImg)

# check the size
print(imgT.shape)
# show image
fig = plt.figure(figsize=(10 ,6))
plt.imshow(imgT)

#%% two convolution kernels

# vertical kernel
VK = np.array([ [1,0,-1],
                [1,0,-1],
                [1,0,-1] ])

# horizontal kernel
HK = np.array([ [ 1, 1, 1],
                [ 0, 0, 0],
                [-1,-1,-1] ])

fig,ax = plt.subplots(2, 2, figsize=(16, 8))

ax[0, 0].imshow(VK)
ax[0, 0].set_title('Vertical kernel')

ax[0, 1].imshow(HK)
ax[0, 1].set_title('Horizontal kernel')

# convolution
# with vertical kernel
convres = convolve2d(imgT, VK, mode='same')
ax[1, 0].imshow(convres, cmap='gray', vmin=0, vmax=0.01)

# with horizontal kernel
convres = convolve2d(imgT, HK,mode='same')
ax[1, 1].imshow(convres, cmap='gray', vmin=0, vmax=0.01)

plt.show()

#%% Pytorch

VK_ts = torch.tensor(VK).view(1, 1, 3, 3).double()
HK_ts = torch.tensor(HK).view(1, 1, 3, 3).double()
img_ts = torch.tensor(imgT).view(1, 1, imgT.shape[0], imgT.shape[1])

print(VK_ts.shape)
print(img_ts.shape)

fig,ax = plt.subplots(2,2,figsize=(16,8))

ax[0, 0].imshow(VK)
ax[0, 0].set_title('Vertical kernel')

ax[0, 1].imshow(HK)
ax[0, 1].set_title('Horizontal kernel')


# convolution
# with vertical kernel
convres = F.conv2d(img_ts, VK_ts)
im = torch.squeeze(convres.detach())
ax[1,0].imshow(im, cmap='gray', vmin=0, vmax=0.01)

convres = F.conv2d(img_ts, HK_ts)
im = torch.squeeze(convres.detach())
ax[1,1].imshow(im, cmap='gray', vmin=0, vmax=0.01)

plt.show()

#%% experiment on size calculation

# params
inChans  = 3 # RGB
imsize   = [64, 64]
outChans = 10
krnSize  = 11 # odd number
stride   = (2, 2)
padding  = 1

# create the instance
c = nn.Conv2d(inChans, outChans, krnSize, stride, padding)

# create an image
img = torch.rand(1, inChans, imsize[0], imsize[1])

# run convolution
resimg = c(img)
empSize = torch.squeeze(resimg).shape

# compute the size 
expectSize = np.array([outChans, 0, 0], dtype=int)
expectSize[1] = np.floor( (imsize[0]+2*padding-krnSize)/stride[0] ) + 1
expectSize[2] = np.floor( (imsize[1]+2*padding-krnSize)/stride[1] ) + 1

# check the size
print(f'Expected size: {expectSize}')
print(f'Empirical size: {list(empSize)}')

