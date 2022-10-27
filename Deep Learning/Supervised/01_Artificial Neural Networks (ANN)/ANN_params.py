# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 00:06:43 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import math
import time
import random
import os
import sympy as sym

# PyTroch
import torch
import torch.nn as nn

# Seaborn
import seaborn as sns

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5 # default 0.8

#%%

# build two models

widenet = nn.Sequential(
    nn.Linear(2,4),  # hidden layer
    nn.Linear(4,3),  # output layer
    )


deepnet = nn.Sequential(
    nn.Linear(1,2),  # hidden layer
    nn.Linear(2,2),  # hidden layer
    nn.Linear(2,3),  # output layer
    )

# print them out to have a look
print(widenet)
print(' ')
print(deepnet)

#%% check out the parameters
for p in deepnet.named_parameters():
  print(p)
  print(' ')

#%% count the number of nodes ( = the number of biases)

# named_parameters() is an iterable that returns the tuple (name,numbers)
numNodesInWide = 0
for p in widenet.named_parameters():
  if 'bias' in p[0]:
    numNodesInWide += len(p[1])

numNodesInDeep = 0
for paramName,paramVect in deepnet.named_parameters():
  if 'bias' in paramName:
    numNodesInDeep += len(paramVect)


print('There are %s nodes in the wide network.' %numNodesInWide)
print('There are %s nodes in the deep network.' %numNodesInDeep)

#%% print just the parameters
for p in widenet.parameters():
  print(p)
  print(' ')
  
#%% count the total number of trainable parameters
nparams = 0
for p in widenet.parameters():
  if p.requires_grad: # if p.requires_grad == true, if requires_grad == false, the parameters are fixed and not be trained
    print('This piece has %s parameters' %p.numel())
    nparams += p.numel()

print('\n\nTotal of %s parameters'%nparams)

#%% list comprehension

nparams = np.sum([ p.numel() for p in widenet.parameters() if p.requires_grad ])
print('Widenet has %s parameters'%nparams)

nparams = np.sum([ p.numel() for p in deepnet.parameters() if p.requires_grad ])
print('Deepnet has %s parameters'%nparams)

#%% print out the model info.
from torchsummary import summary
summary(widenet,(1,2))