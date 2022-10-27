# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 00:19:16 2022

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

# Scipy
import scipy.stats as stats

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)

#%% Generate random data

# parameters
n1 = 30   # n samples dataset 1
n2 = 40   # n samples dataset 2
mu1 = 1   # mean offset dataset 1
mu2 = 2   # mean offset in dataset 2

# generate the data
data1 = mu1 + np.random.randn(n1)
data2 = mu2 + np.random.randn(n2)

# plot
plt.plot(np.zeros(n1),data1,'ro',markerfacecolor='w',markersize=5)
plt.plot(np.ones(n2), data2,'bs',markerfacecolor='w',markersize=5)
plt.xlim([-1,2])
plt.xticks([0,1],labels=['Group 1','Group 2'])
plt.show()

#%% t-test

# _ind = independent samples
t, p = stats.ttest_ind(data1, data2)
print(t) # -6.467971782048848 , minus just order of the inputs, arbitrary
print(p) # 1.2713993335986364e-08, dataset 2 has statistically significant larger mean than dataset 1
# t, p = stats.ttest_ind(data2, data1)

# t-test results plot
fig = plt.figure(figsize=(10,4))
plt.rcParams.update({'font.size':12}) # change the font size

# for visualization purpose
plt.plot(0+np.random.randn(n1)/15, data1, 'ro',markerfacecolor='w',markersize=14)
plt.plot(1+np.random.randn(n2)/15, data2, 'bs',markerfacecolor='w',markersize=14)
plt.xlim([-1,2])
plt.xticks([0,1],labels=['Group 1','Group 2'])

# set the title to include the t-value and p-value
plt.title(f't = {t:.2f}, p={p:.3f}')

plt.show()