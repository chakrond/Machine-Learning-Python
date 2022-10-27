# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 13:32:04 2022

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

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10

#%% Funciton

def line_fun(m):
    N = 50
    x = torch.randn(N, 1)
    y = m*x + torch.randn(N, 1)/2
    return x, y

#%% Generate random data

# test the funciton
m = 1
x_0, y_0 = line_fun(m)
model_m = np.linspace(-2, 2, 21)

# plot data
plt.figure(0)
plt.plot(x_0, y_0, 's')
plt.show()

#%% Model instance & layers

def ANNreg_model(x, y):
    
    epochs = 500
    losses = torch.zeros(epochs)
    
    ANNreg = nn.Sequential(
        nn.Linear(1, 1),  # input layer nn.Linear(in_features=1, out_features=1)
        nn.ReLU(),       # activation function
        nn.Linear(1, 1)   # output layer(input=1, output=1)
        )
    
    #% Metaparams
    
    # Learning rate
    learning_rate = .05
    
    # loss function
    loss_fun = nn.MSELoss()
    
    # optimizer (the flavor of gradient descent to implement)
    optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learning_rate)
        
    for epoch in range(epochs):
            
        # forward pass
        yHat = ANNreg(x) # predicted result from model
          
        # compute loss
        loss = loss_fun(yHat, y) # y = target/true value
        losses[epoch] = loss # store loss in every epoch
        
        # backprop
        optimizer.zero_grad() # set derivative gradient of the model to be zero
        loss.backward() # back propagation on the computed losses
        optimizer.step() # stochastic gradient
        
    # make prediction
    predictions = ANNreg(x_in)
    
    return predictions, losses

#%% Train model

# train the model
n_Expm = 50
final_loss_model = torch.zeros(len(model_m), n_Expm)
corrcoef_model = torch.zeros(len(model_m), n_Expm)


for (i, m) in enumerate(model_m):
    
    for n in range(n_Expm):
        # generate data
        x_in, y_true = line_fun(m)
        
        # get prediciton and losses
        yHat, losses = ANNreg_model(x_in, y_true)
        
        # store final loss of each model
        final_loss_model[i, n] = losses[-1]
        
        # Pearson’s correlation
        corrcoef_model[i, n] = np.corrcoef(y_true.T.detach(), yHat.T.detach())[0, 1]

# assign 0 to 'nan' index in Pearson’s correlation coefficients
corrcoef_model[np.isnan(corrcoef_model).bool()] = 0

  
#%% Plot

plt.figure(1)

# Average loss over experiments
plt.subplot(1, 2, 1)
# plt.plot(model_m, losses[:, -1].detach(), 'o', markerfacecolor='w', linewidth=0.1)
plt.plot(model_m, np.mean(final_loss_model.detach().numpy(), axis=1), 'o-') 
plt.xlabel('Slope')
plt.ylabel('Loss')
plt.title('Final Loss for each slope')
plt.show()

# Average correlation coefficients over experiments
plt.subplot(1, 2, 2)
plt.plot(model_m, np.mean(corrcoef_model.detach().numpy(), axis=1), 'ro-')
plt.xlabel('Slope')
plt.ylabel('Correlation Coefficients')
plt.title('Model Performance')
# plt.legend(['True', 'Prediction'])
plt.show()