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

# Seaborn
import seaborn as sns

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10

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

#%% Model instance & layers

def ANNclf_model(data, y_true, lr):
    
    epochs = 1000
    losses = torch.zeros(epochs)
    acc_ep = torch.zeros(epochs)
    
    ANNclf = nn.Sequential(
        nn.Linear(4, 64),   # input layer nn.Linear(in_features, out_features)
        nn.ReLU(),          # activation function
        nn.Linear(64, 64),  # hidden layers
        nn.ReLU(),          # activation function
        nn.Linear(64, 3),   # output layer(input=64, output=3)
                            
        )
    
    # ---------------------------------------    
    # Metaparams
    # ---------------------------------------
    
    # Learning rate
    learning_rate = lr
    
    # loss function 
    loss_fun = nn.CrossEntropyLoss() # already include computing Softmax activation function at the output layer
    
    # optimizer
    optimizer = torch.optim.SGD(ANNclf.parameters(), lr=learning_rate) # stochastic gradient
     
    # ---------------------------------------
    
    # begin training the model
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
        
        # compute accuracy per epoch
        labels_pred_ep = torch.argmax(y_pred, axis=1) == y_true            # pick the highest probability and compare to true labels
        acc_epoch = 100*torch.sum(labels_pred_ep.float()) / len(labels_pred_ep)   # correct percentage
        acc_ep[epoch] = acc_epoch


    # Test on training set
    # Make prediction
    predictions = ANNclf(data)
    
    # Model Accuracy
    labels_pred = torch.argmax(predictions, axis=1) == y_true
    total_acc = 100*torch.sum(labels_pred.float()) / len(labels_pred)
    
    return y_pred, predictions, losses, total_acc, acc_ep, ANNclf

#%% Create/Train model

# learning rate
model_lr = np.linspace(0.001, 0.1, 40)

# store result
epochs = 1000
n_exprms = 5
res_losses = np.zeros( (len(model_lr), epochs, n_exprms) )
res_acc_ep = np.zeros( (len(model_lr), epochs, n_exprms) )
res_acc = np.zeros( (len(model_lr), n_exprms) )
res_y_pred = np.zeros( (len(labels), len(np.unique(labels)), len(model_lr)) )
list_expms_y_pred = []

# run experiments
for e in range(n_exprms):
    
    list_y_pred = []
    
    for (i, lr) in enumerate(model_lr):
    
        model_y_pred, model_predicitons, model_losses, model_acc, model_acc_ep, _  = ANNclf_model(data, labels, lr)
        res_losses[i, :, e] = model_losses.detach().numpy()
        res_acc_ep[i, :, e] = model_acc_ep.detach().numpy()
        res_y_pred[:, :, i] = model_y_pred.detach().numpy()
        list_y_pred.append(res_y_pred)
        res_acc[i, e] = model_acc.detach().numpy()
    
    list_expms_y_pred.append(list_y_pred)

# Softmax
sm = nn.Softmax(dim=1)

# average over learning rate and experiments
mean_y_pred = np.mean(np.mean(np.mean(list_expms_y_pred, axis=0), axis=0), axis=2)

#%% plot model performance

# plot final losses
fig = plt.figure()
fig.suptitle(f'Overall Performance of {n_exprms} Models, {len(model_lr)} Learning Rate, {epochs} epochs', fontweight='bold')

plt.subplot(2, 2, 1)
plt.plot(model_lr, np.mean(res_losses, axis=2)[:, -1], 'o-', markerfacecolor='w', linewidth=0.5)
plt.xlabel('Learning Rate')
plt.ylabel('Final Loss')
plt.title('Final Loss vs Learning Rate')
plt.legend(['Final Loss'])
plt.show()

# plot accuracy
# plt.figure()
plt.subplot(2, 2, 2)
plt.plot(model_lr, np.mean(res_acc, axis=1), 'bo-', markerfacecolor='w', linewidth=0.5)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Learning Rate')
plt.legend(['Accuracy'])
plt.show()

# plot losses-epoch by model
# plt.figure()
plt.subplot(2, 2, 3)
plt.plot(np.mean(res_losses, axis=2).T, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses vs Learning Rate')
plt.show()

# plot acc-epoch by model
# plt.figure()
plt.subplot(2, 2, 4)
plt.plot(np.mean(res_acc_ep, axis=2).T, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Learning Rate')
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

