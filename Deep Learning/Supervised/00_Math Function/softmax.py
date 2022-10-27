# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 03:28:07 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import random
import os

# PyTorch
# import torch
# import torch.nn as nn

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)

#%% Softmax

# number list
z = np.arange(0, 26)

# softmax result
num = np.exp(z)
s_num = np.sum(num)
sigma = num / s_num

# random integers
a = np.random.randint(-5, high=15, size=25)
a_num = np.exp(a)
a_sum = np.sum(a_num)
a_sigma = a_num / a_sum

# Plot
fig = plt.figure(1)
# fig.suptitle(f'UCB vs Thompson - iteration = {N}',fontweight ="bold")

plt.subplot(1, 3, 1)
plt.plot(z, sigma, 'bo')
plt.title(f'Softmax, sum = {sum(sigma)}')
plt.xlabel('Numbers')
plt.ylabel('Softmax (Sigma)')

plt.subplot(1, 3, 2)
plt.plot(a, a_sigma, 'ko')
plt.title(f'Softmax, sum = {sum(a_sigma)}')
plt.xlabel('Numbers')
plt.ylabel('Softmax (Sigma)')

plt.subplot(1, 3, 3)
plt.plot(a, a_sigma, 'ko')
plt.title(f'Softmax - Log Scale, sum = {sum(a_sigma)}')
plt.xlabel('Numbers')
plt.ylabel('Softmax (Sigma) - Log Scale')
plt.yscale('log')

