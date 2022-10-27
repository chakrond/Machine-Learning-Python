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

#%% Generate random data

N = 30
x = torch.randn(N, 1)
y = x + torch.randn(N, 1)/2

# plot
plt.figure(0)
plt.plot(x, y, 's')
plt.show()

#%% Model instance & layers

ANNreg = nn.Sequential(
    nn.Linear(1, 1),  # input layer nn.Linear(in_features=1, out_features=1)
    nn.ReLU(),       # activation function
    nn.Linear(1, 1)   # output layer(input=1, output=1)
    )

ANNreg

#%% Metaparams

# Learning rate
learning_rate = .05

# loss function
loss_fun = nn.MSELoss()

# optimizer (the flavor of gradient descent to implement)
optimFun = getattr(torch.optim, 'Adam')
thisdict = {
  'lr': 1,
  'weight_decay': 0.5,
}

del thisdict['lr']

# optimizer = optimFun(ANNreg.parameters(), lr=learning_rate, weight_decay=0.01, momentum=0.85)
optimizer = optimFun(ANNreg.parameters(), **thisdict)

#%% Train model

# train the model
epochs = 200
losses = torch.zeros(epochs)

for epoch in range(epochs):

  # forward pass
  yHat = ANNreg(x) # predicted result from model

  # compute loss
  loss = loss_fun(yHat, y) # y = target/true value
  losses[epoch] = loss

  # backprop
  optimizer.zero_grad() # set derivative gradient of the model to be zero
  loss.backward() # back propagation on the computed losses
  optimizer.step() # stochastic gradient
  
#%% Observe the loss

# compute losses

predictions = ANNreg(x)

# final loss - Mean Squared Error(MSE)
test_loss = (predictions - y).pow(2).mean()

plt.figure(1)
plt.plot(losses.detach(), 'o', markerfacecolor='w', linewidth=0.1)
plt.plot(epochs, test_loss.detach(),'ro') # test_loss.detach() or test_loss.item()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final test loss = %g' %test_loss.item())
plt.show()

plt.figure(2)
plt.plot(x, y, 'o', markerfacecolor='w', linewidth=0.1)
plt.plot(x, predictions.detach(),'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('True vs Prediciton')
plt.legend(['True', 'Prediction'])
plt.show()