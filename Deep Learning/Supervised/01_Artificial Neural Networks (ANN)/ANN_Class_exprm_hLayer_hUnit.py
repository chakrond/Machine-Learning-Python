# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 01:49:06 2022

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

#%% ANN Class

class ANN_Class(nn.Module):
    
    # Inheritance from parents class
    def __init__(self, feature_in, feature_out, n_hUnit, n_hLayer):
        super().__init__()
        # super(ANN_Class, self).__init__()
        
        # ---------------------------------------    
        # Layer structure
        # ---------------------------------------
        self.layers = nn.ModuleDict() # store layers
        
        self.n_hLayer = n_hLayer # number of hidden layers
        
        self.layers['input'] = nn.Linear(feature_in, n_hUnit) # input layer
        
        for i in range(n_hLayer):
            self.layers[f'hidden{i}'] = nn.Linear(n_hUnit, n_hUnit) # hidden layers
        
        self.layers['output'] = nn.Linear(n_hUnit, feature_out) # output layer
        
        # ---------------------------------------
    
    # forward pass
    def forward(self, x):
        # input layer
        x = self.layers['input'](x)
    
        # hidden layers
        for i in range(self.n_hLayer):
          x = F.relu( self.layers[f'hidden{i}'](x) )
        
        # return output layer
        x = self.layers['output'](x)
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
    
        # total number of trainable parameters in the model
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        return y_pred, losses, acc_ep, n_params


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
model_params_0 = np.arange(4, 101, 3) # number of unit per layer
model_params_1 = range(1,6) # number of hidden layers
model_params_name = ['Number of Hidden Layer', 'Number of Unit per Layer']

# model parameters
lr = 0.01

# store result
epochs = 2500
res_losses = np.zeros( (len(model_params_1), epochs, len(model_params_0)) )
res_acc_ep = np.zeros( (len(model_params_1), epochs, len(model_params_0)) )
res_acc = np.zeros( (len(model_params_1), len(model_params_0)) )
res_y_pred = np.zeros( (len(labels), len(np.unique(labels)), len(model_params_1)) )
list_y_pred_params_0 = []
nParams = np.zeros( (len(model_params_1), len(model_params_0)) )

# run experiments
for (e, param_0) in enumerate(model_params_0):
    
    list_y_pred_params_1 = []
    
    for (i, param_1) in enumerate(model_params_1):
    
        # Model class instance
        ANNclf_model = ANN_Class(feature_in=4, feature_out=3, n_hUnit=param_0, n_hLayer=param_1)
        
        model_y_pred, model_losses, model_acc_ep, model_nParams  = ANNclf_model.train(data, labels, epochs=epochs, p_lr=.01)
        model_predicitons, model_acc = ANNclf_model.predict(data, labels)
        res_losses[i, :, e] = model_losses.detach().numpy()
        res_acc_ep[i, :, e] = model_acc_ep.detach().numpy()
        res_y_pred[:, :, i] = model_y_pred.detach().numpy()
        list_y_pred_params_1.append(res_y_pred)
        res_acc[i, e] = model_acc.detach().numpy()
        nParams[i, e] = model_nParams
    
    list_y_pred_params_0.append(list_y_pred_params_1)

# Softmax
sm = nn.Softmax(dim=1)

# average over focus hyperparams and experiments
mean_y_pred = np.mean(np.mean(np.mean(list_y_pred_params_0, axis=0), axis=0), axis=2)

#%% plot model performance

# plot final losses
fig = plt.figure()
fig.tight_layout()
fig.suptitle(f'Overall Performance', fontweight='bold')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
plt.subplot(2, 2, 1)
plt.plot(model_params_0, np.mean(res_losses, axis=1).T, 'o-', markerfacecolor='w', linewidth=0.5)
plt.xlabel(f'{model_params_name[1]}')
plt.ylabel('Final Loss')
plt.title(f'Final Loss by Learning vs {model_params_name[1]}')
plt.legend([f'{i} Hidden Layers'for i in model_params_1])
plt.show()

# plot acc-epoch by model
# plt.figure()
plt.subplot(2, 2, 2)
plt.plot(model_params_0, np.mean(res_acc_ep, axis=1).T, 'o-', markerfacecolor=None, linewidth=1)
plt.xlabel(f'{model_params_name[1]}')
plt.ylabel('Accuracy')
plt.title(f'Accuracy by Learning vs {model_params_name[1]}')
plt.legend([f'{i} Hidden Layers'for i in model_params_1])
plt.show()

# # plot accuracy
# # plt.figure()
# plt.subplot(2, 2, 2)
# plt.plot(model_params_0, res_acc.T, 'o-', markerfacecolor='w', linewidth=0.5)
# plt.xlabel(f'{model_params_name[1]}')
# plt.ylabel('Accuracy')
# plt.title(f'Prediction Accuracy vs {model_params_name[1]}')
# plt.legend(model_params_1)
# plt.show()

# # plot losses-epoch by model
# # plt.figure()
# plt.subplot(2, 2, 3)
# plt.plot(model_params_0, np.mean(res_losses, axis=1).T, 'o-', markerfacecolor=None, linewidth=1)
# plt.xlabel(f'{model_params_name[1]}')
# plt.ylabel('Loss')
# plt.title(f'Losses vs {model_params_name[1]}')
# plt.legend(model_params_1)
# plt.show()

# # plot losses-epoch by model
# # plt.figure()
plt.subplot(2, 2, 3)
plt.plot(np.arange(1, epochs+1), np.mean(res_losses, axis=2).T, linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses vs Epoch')
plt.legend([f'{i} Hidden Layers'for i in model_params_1])
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

#%% plot parameters coerrlation

# vectorize
x = nParams.flatten()
y = res_acc.flatten()

# correlation between them
r = np.corrcoef(x, y)[0, 1]

# scatter plot
plt.plot(x, y, 'o')
plt.xlabel('Number of parameters')
plt.ylabel('Accuracy')
plt.title('Correlation: r=' + str(np.round(r, 3)))
plt.show()
