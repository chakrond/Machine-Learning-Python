# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 05:54:57 2022

@author: Chakron.D
"""

# %% Importing the libraries

# numpy
import numpy as np
import copy
import math

# PyTroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Loss functions

# abs mean
class mLoss(nn.Module):

    # Inheritance from parents class
    def __init__(self):
        super().__init__()

    def forward(self, yHat, yTarget):
        
        res = torch.mean(torch.abs(yTarget - yHat))
            
        return res


# confunction loss
class ConjLoss(nn.Module):

    # Inheritance from parents class
    def __init__(self):
        super().__init__()

    def forward(self, yHat, yTarget):
        # MSE term + mean
        res = torch.mean((yTarget - yHat)**2) + torch.abs(torch.mean(yHat))
            
        return res
    

# correlation loss
class CorrLoss(nn.Module):

    # Inheritance from parents class
    def __init__(self):
        super().__init__()

    def forward(self, yHat, yTarget):
        
        u_yHat = torch.mean(yHat)
        u_yTar = torch.mean(yTarget)
        std_yHat = torch.std(yHat)
        std_yTar = torch.std(yTarget)
        
        res = -1*torch.sum( (yHat-u_yHat)*(yTarget-u_yTar) ) / ( (yHat.numel()-1)*std_yHat*std_yTar )
        
        return res