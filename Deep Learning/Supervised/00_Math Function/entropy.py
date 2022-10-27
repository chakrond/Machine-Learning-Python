# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 18:56:56 2022

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

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)

#%% entropy

# Probability
x = [0.25, 0.65, 0.1] # all possible case sum == 1

H = 0
for p in x:
    H += -p*np.log(p)

print(f'Entropy: {H}')

#%% Binary cross-entropy (2 possible cases)
p = 0.25
H = -( p*np.log(p) + (1-p)*np.log(1-p) )
print(f'binary cross-entropy: {H}')

#%% cross entropy between p and q
p = [1, 0] # sum == 1, true class
q = [0.25, 0.75] # sum == 1, result from model

H = 0
for i in range(len(p)):
    H += -p[i]*np.log(q[i])
    
print(f'cross-entropy between p and q: {H}')

#%% Pytroch

import torch
import torch.nn.functional as F

p_tensor = torch.Tensor(p) # true  class
q_tensor = torch.Tensor(q) # model prediction

H = F.binary_cross_entropy(input=q_tensor, target=p_tensor)
print(f'[Troch] - cross-entropy between p and q: {H}')