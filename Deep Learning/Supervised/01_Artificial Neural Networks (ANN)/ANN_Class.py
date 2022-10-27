# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 00:38:39 2022

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
import torch.nn.functional as F

# Seaborn
import seaborn as sns

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5 # default 0.8

#%%

class ANN_Class(nn.Module):
    
    # Inheritance from parents class
    def __init__(self, ):
        super(ANN_Class, self).__init__()
        
        # ---------------------------------------    
        # Metaparams
        # ---------------------------------------
    
        self.n_hLayer = 64 # hidden layers
        self.input = nn.Linear(4, self.n_hLayer) # input layer
        self.output = nn.Linear(self.n_hLayer, 3) # output layer
        
        # ---------------------------------------
    
    # forward pass
    def forward(self, x):
    
      # pass through the input layer
      x = self.input(x)
    
      # apply relu
      x = F.relu(x)
    
      # output layer
      x = self.output(x)
      # x = torch.sigmoid(x)
    
      return x
    
    
    def train(self, data, y_true, epochs=1000, p_lr=0.01):
    
        # ---------------------------------------    
        # Metaparams
        # ---------------------------------------
        # Learning rate
        learning_rate = p_lr
        
        # loss function 
        loss_fun = nn.CrossEntropyLoss() # already include computing Softmax activation function at the output layer
        # loss_fun = nn.BCELoss()
        # optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate) # stochastic gradient
    
        # ---------------------------------------
    
    
        # store data
        losses = torch.zeros(epochs)
        acc_ep = torch.zeros(epochs)    
    
        # begin training the model
        for epoch in range(epochs):
                
            # forward pass
            y_pred = self.forward(data) # predicted result from model
              
            # compute loss
            loss = loss_fun(y_pred, y_true) # y = target/true value
            losses[epoch] = loss # store loss in every epoch
            
            # backprop
            optimizer.zero_grad() # set derivative gradient of the model to be zero
            loss.backward() # back propagation on the computed losses
            optimizer.step() # stochastic gradient
            
            # compute accuracy per epoch
            labels_pred_ep = torch.argmax(y_pred, axis=1) == y_true  # pick the highest probability and compare to true labels
            acc_epoch = 100*torch.sum(labels_pred_ep.float()) / len(labels_pred_ep)   # correct percentage
            acc_ep[epoch] = acc_epoch
    
        return y_pred, losses, acc_ep


    def predict(self, data, y_true):
        
        # Make prediction
        predictions = self.forward(data)
        
        # Model Accuracy
        labels_pred = torch.argmax(predictions, axis=1) == y_true
        total_acc = 100*torch.sum(labels_pred.float()) / len(labels_pred)
    
        return predictions, total_acc
    
#%% Import dataset

# Iris dataset
iris = sns.load_dataset('iris')

#%% data preprocessing

# convert from pandas dataframe to tensor
data = torch.tensor( iris[iris.columns[0:4]].values ).float()

# transform species to number
labels = torch.zeros(len(data), dtype=torch.long)
# labels[iris.species=='setosa'] = 0 
labels[iris.species=='versicolor'] = 1
labels[iris.species=='virginica'] = 2

# plots data
# sns.pairplot(iris, hue='species')
# plt.show()

#%% Create/Train model

# experiment parameters
model_params = np.linspace(0.001, 0.1, 40) # learning rate
# model_params = np.arange(1, 129) # hidden layers
model_params_name = ['Learning Rate']

# model parameters
lr = 0.01

# store result
epochs = 1000
n_exprms = 5
res_losses = np.zeros( (len(model_params), epochs, n_exprms) )
res_acc_ep = np.zeros( (len(model_params), epochs, n_exprms) )
res_acc = np.zeros( (len(model_params), n_exprms) )
res_y_pred = np.zeros( (len(labels), len(np.unique(labels)), len(model_params)) )
list_expms_y_pred = []

# run experiments
for e in range(n_exprms):
    
    list_y_pred = []
    
    for (i, param) in enumerate(model_params):
    
        # Model class instance
        ANNclf_model = ANN_Class()
        
        model_y_pred, model_losses, model_acc_ep  = ANNclf_model.train(data, labels, epochs=epochs, p_lr=param)
        model_predicitons, model_acc = ANNclf_model.predict(data, labels)
        res_losses[i, :, e] = model_losses.detach().numpy()
        res_acc_ep[i, :, e] = model_acc_ep.detach().numpy()
        res_y_pred[:, :, i] = model_y_pred.detach().numpy()
        list_y_pred.append(res_y_pred)
        res_acc[i, e] = model_acc.detach().numpy()
    
    list_expms_y_pred.append(list_y_pred)

# Softmax
sm = nn.Softmax(dim=1)

# average over focus hyperparams and experiments
mean_y_pred = np.mean(np.mean(np.mean(list_expms_y_pred, axis=0), axis=0), axis=2)


#%% plot model performance

# plot final losses
fig = plt.figure()
fig.tight_layout()
fig.suptitle(f'Overall Performance of {n_exprms} Models, {len(model_params)} {model_params_name[0]}, {epochs} epochs', fontweight='bold')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
plt.subplot(2, 2, 1)
plt.plot(model_params, np.mean(res_losses, axis=2)[:, -1], 'o-', markerfacecolor='w', linewidth=0.5)
plt.xlabel(f'{model_params_name[0]}')
plt.ylabel('Final Loss')
plt.title(f'Final Loss vs {model_params_name[0]}')
plt.legend(['Final Loss'])
plt.show()

# plot accuracy
# plt.figure()
plt.subplot(2, 2, 2)
plt.plot(model_params, np.mean(res_acc, axis=1), 'bo-', markerfacecolor='w', linewidth=0.5)
plt.xlabel(f'{model_params_name[0]}')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs {model_params_name[0]}')
plt.legend(['Accuracy'])
plt.show()

# plot losses-epoch by model
# plt.figure()
plt.subplot(2, 2, 3)
plt.plot(np.mean(res_losses, axis=2).T, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Losses vs {model_params_name[0]}')
plt.show()

# plot acc-epoch by model
# plt.figure()
plt.subplot(2, 2, 4)
plt.plot(np.mean(res_acc_ep, axis=2).T, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs {model_params_name[0]}')
plt.show()

#%% plot prediciton results


fig = plt.figure()
fig.suptitle(f'Overall performance of model prediction', fontweight='bold')

plt.subplot(1, 2, 1)
plt.plot(sm(torch.tensor(mean_y_pred)) ,'s-',markerfacecolor='w', alpha=0.8)
plt.xlabel('Stimulus number')
plt.ylabel('Probability (Softmax)')
plt.legend(['setosa','versicolor','virginica'])
plt.show()

plt.subplot(1, 2, 2)
plt.plot(mean_y_pred ,'o-',markerfacecolor='w', alpha=0.8)
plt.xlabel('Stimulus number')
plt.ylabel('Probability (raw)')
plt.legend(['setosa','versicolor','virginica'])
plt.show()
