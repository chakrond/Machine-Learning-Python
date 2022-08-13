# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 04:41:12 2022

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

#%% Generate the data

nPerClust = 100
blur = 1

A = [1, 1]
B = [5, 1]

# generate data
a = [ A[0] + (np.random.randn(nPerClust) * blur) , A[1] + (np.random.randn(nPerClust) * blur) ]
b = [ B[0] + (np.random.randn(nPerClust) * blur) , B[1] + (np.random.randn(nPerClust) * blur) ]

# true labels
labels_np = np.vstack( (np.zeros((nPerClust, 1)), np.ones((nPerClust, 1))) ) # vertical concat

# concatanate into a matrix
data_np = np.hstack((a, b)).T # horizontal concat

# convert to a pytorch tensor
data = torch.tensor(data_np).float()
labels = torch.tensor(labels_np).float()

# show the data
plt.figure()
plt.plot(data[labels.ravel()==0, 0], data[labels.ravel()==0, 1], 'bs')
plt.plot(data[labels.ravel()==1, 0], data[labels.ravel()==1, 1], 'ko')
plt.show()

#%% Model instance & layers

def ANNclass_model(data, y_true):
    
    epochs = 1000
    losses = torch.zeros(epochs)
    
    ANNclf = nn.Sequential(
        nn.Linear(2, 1),  # input layer nn.Linear(in_features=2, out_features=1)
        nn.ReLU(),       # activation function
        nn.Linear(1, 1),   # output layer(input=1, output=1)
        nn.Sigmoid()
        )
        
    #% Metaparams
    
    # Learning rate
    learning_rate = 0.01
    
    # loss function 
    loss_fun = nn.BCELoss() # binary classification error(BCE)
    
    # optimizer 
    optimizer = torch.optim.SGD(ANNclf.parameters(), lr=learning_rate)
        
    for epoch in range(epochs):
            
        # forward pass
        y_pred = ANNclf(data) # predicted result from model
          
        # compute loss
        loss = loss_fun(y_pred, y_true) # y = target/true value
        losses[epoch] = loss # store loss in every epoch
        
        # backprop
        optimizer.zero_grad() # set derivative gradient of the model to be zero
        loss.backward() # back propagation on the computed losses
        optimizer.step() # stochastic gradient
        
    # make prediction
    predictions = ANNclf(data)
    
    return predictions, losses, ANNclf

#%% Create/Train model

model_predicitons, model_losses, model_clf = ANNclass_model(data, labels)

#%% plot

# plot losses
plt.figure()
plt.plot(model_losses.detach(), '.', markerfacecolor='w', linewidth=0.1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# plot prediction labels
label_pred = model_predicitons > 0.5
label_pred_err = label_pred != labels
pred_acc = 100*sum(~label_pred_err)/len(label_pred)

plt.figure()

# error
plt.plot(data[label_pred_err.ravel(), 0] ,data[label_pred_err.ravel(), 1], 'rx', markersize=12, markeredgewidth=3)

# predicted
plt.plot(data[~label_pred.ravel(), 0], data[~label_pred.ravel(), 1], 'bs')
plt.plot(data[label_pred.ravel(), 0], data[label_pred.ravel(), 1], 'ko')


#%% Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = data, labels
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.1),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.1))

# data_tensor = torch.tensor((np.array([X1.ravel(), X2.ravel()]).T)).float()
# Z = model_clf(data_tensor).detach().numpy()

# plt.figure()
# plt.contourf(X1, X2, Z.reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())

# for (i, j) in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[:, 0].reshape((len(X_set), 1))[y_set == j], X_set[:, 1].reshape((len(X_set), 1))[y_set == j], c = ListedColormap(('red', 'green'))(i), label = j)

