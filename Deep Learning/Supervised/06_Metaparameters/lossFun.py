# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 01:49:06 2022

@author: Chakron.D
"""

#%% Importing the libraries

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

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5 # default 0.8

#%% MSE

# loss function
lossfunMSE = nn.MSELoss()

# create predictions and real answer
yHat = torch.linspace(-2, 2, 101)
yTrue = torch.tensor(.5)

# compute MSE loss function
L = np.zeros(101)
for i, yy in enumerate(yHat):
  L[i] = lossfunMSE(yy, yTrue)

plt.plot(yHat, L, label='Loss')
plt.plot([yTrue, yTrue], [0, np.max(L)], 'r--', label='True value')
plt.xlabel('Predicted value')
plt.legend()
plt.show()

#%% Binary cross-entropy

# loss function
lossfunBCE = nn.BCELoss()

# create predictions and real answer
yHat = torch.linspace(.001,.999,101)
y1 = torch.tensor(0.)
y2 = torch.tensor(1.)

# compute MSE loss function
L = np.zeros((101,2))
for i,yy in enumerate(yHat):
  L[i,0] = lossfunBCE(yy,y1) # 0 is the correct answer
  L[i,1] = lossfunBCE(yy,y2) # 1 is the correct answer

plt.plot(yHat,L)
plt.xlabel('Predicted value')
plt.ylabel('Loss')
plt.legend(['correct=0','correct=1'])
# plt.yscale('log')
plt.show()


# "raw" output of a model
yHat = torch.tensor(2.)

# convert to prob via sigmoid
sig = nn.Sigmoid()
print(lossfunBCE( sig(yHat) ,y2))


# recommended way
lossfunBCE = nn.BCEWithLogitsLoss()
yHat = torch.tensor(2.)
print(lossfunBCE(yHat,y2))

#%% Categorical cross-entropy

# loss function
lossfunCCE = nn.CrossEntropyLoss()

# vector of output layer (pre-softmax)
yHat = torch.tensor([[1.,4,3]])

for i in range(3):
  correctAnswer = torch.tensor([i])
  thisloss = lossfunCCE(yHat,correctAnswer).item()
  print( 'Loss when correct answer is %g: %g' %(i,thisloss) )
  
# compare raw, softmax, and log-softmax outputs
sm = nn.Softmax(dim=1)
log_sm = nn.LogSoftmax(dim=1)

yHat_sm = sm(yHat)
yHat_logsm = log_sm(yHat)

# print them
print(yHat)
print(yHat_sm)
print(yHat_logsm)

#%% create own loss function

class myLoss(nn.Module): # inherent info from nn.Module
  def __init__(self):
    super().__init__()
      
  def forward(self,x,y):
    loss = torch.abs(x-y)
    return loss

# test
lfun = myLoss()
loss = lfun(torch.tensor(4),torch.tensor(5.2))
print(f'Loss is {loss}')

