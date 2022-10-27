# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 01:49:06 2022

@author: Chakron.D
"""

#%% Importing the libraries

# python
import math
import time
import random
import os

# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

# panda
import pandas as pd

# Symbol python
import sympy as sym

# PyTroch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Seaborn
import seaborn as sns

# sckitlearn
from sklearn.model_selection import train_test_split


cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5 # default 0.8

#%% ANN Class

class ANN_Class(nn.Module):
    
    # Inheritance from parents class
    def __init__(self, feature_in, feature_out, n_hUnit, n_hLayer, dropout_rate=0.5, learning_rate=0.01, p_lambda=0.01):
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
        # Parameters
        # ---------------------------------------
        # Dropout rate
        self.dr = dropout_rate
        # Learning rate
        self.lr = learning_rate
        
        # ---------------------------------------
        # Optimizer
        # ---------------------------------------
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=p_lambda) # stochastic gradient
        
    
    # forward pass
    def forward(self, x):
        
        # input layer
        x = self.layers['input'](x)
    
        # dropout after input layer
        x = F.dropout(x, p=self.dr, training=self.training) # training=self.training means to turn off during eval mode
    
        # hidden layers
        for i in range(self.n_hLayer):
            
            # hidden layer
            x = F.relu( self.layers[f'hidden{i}'](x) )
            # dropout
            x = F.dropout(x, p=self.dr, training=self.training)
        
        # return output layer
        x = self.layers['output'](x)
        return x
    
    
    def trainModel(self, 
              DataLoader_train,
              DataLoader_test,
              epochs=1000,
              batch_size=12,
              loss_function=None
              ):
    
        # ---------------------------------------    
        # Metaparams
        # ---------------------------------------
        self.loss_func = loss_function
        
        # loss function 
        if loss_function == 'cross-entropy':
            loss_fun = nn.CrossEntropyLoss() # already include computing Softmax activation function at the output layer
        
        if loss_function == 'binary':
            loss_fun = nn.BCEWithLogitsLoss() # already combines a Sigmoid layer
        
        # ---------------------------------------
        # store results
        # ---------------------------------------
        
        # train set
        
        # epoch
        losses_ep_train = torch.zeros(epochs) 
        acc_ep_train = torch.zeros(epochs)
        

        # test set
        acc_ep_test = torch.zeros(epochs)
    
        # begin training the model
        for epoch in range(epochs):
                
            #  switch training on, and implement dropout
            self.train()
            
            # batch
            losses_batch_train = torch.zeros(len(DataLoader_train))  
            acc_batch_train = torch.zeros(len(DataLoader_train))  
            
            batch = 0
            for X_batch_train, y_batch_true_train in DataLoader_train:
                
                # forward pass
                y_pred_train = self.forward(X_batch_train) # predicted result from model
                  
                # compute loss
                loss = loss_fun(y_pred_train, y_batch_true_train) # y = target/true value
                losses_batch_train[batch] = loss # store loss in every epoch
                
                # backpropagation
                self.optimizer.zero_grad() # set derivative gradient of the model to be zero
                loss.backward() # back propagation on the computed losses
                self.optimizer.step() # stochastic gradient
            
                # compute accuracy per batch of training set
                if loss_function == 'cross-entropy':
                    labels_pred_batch_train = torch.argmax(y_pred_train, axis=1) == y_batch_true_train  # pick the highest probability and compare to true labels
                
                if loss_function == 'binary': 
                    labels_pred_batch_train = ((y_pred_train>0.5) == y_batch_true_train).float()  # pick the probability>0.5 and compare to true labels # 100*torch.mean(((predictions>0.5) == labels).float())
                
                acc_ba_train = 100*torch.sum(labels_pred_batch_train.float()) / len(labels_pred_batch_train)   # correct percentage
                acc_batch_train[batch] = acc_ba_train 
                
                # batch increment
                batch += 1
                # ----------------------------------------  
                
                
            # average batch losses&acc
            losses_ep_train[epoch] = torch.mean(losses_batch_train)
            acc_ep_train[epoch] = torch.mean(acc_batch_train)
                
             
            # compute accuracy per epoch of test set
            X_ep_test, y_ep_true_test = next(iter(DataLoader_test))
            _, acc_epoch_test =  self.predict(X_ep_test, y_ep_true_test)
            acc_ep_test[epoch] = acc_epoch_test
            # ---------------------------------------- 
    
        # total number of trainable parameters in the model
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # ---------------------------------------- 
        
        return y_pred_train, losses_ep_train, acc_ep_train, acc_ep_test, n_params


    def predict(self, data, y_true):
        
        # Make prediction
        self.eval() # switch training off and no dropout during this mode
        predictions = self.forward(data)
        
        # Model Accuracy
        if self.loss_func == 'cross-entropy':
            labels_pred = torch.argmax(predictions, axis=1) == y_true
    
        if self.loss_func == 'binary': 
            labels_pred = ((predictions>0.5) == y_true).float()
            
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

#%% train/test dataset
model_test_size = 0.2
batch_size = 64

# split to find array length
data_train,data_test, labels_train,labels_test = train_test_split(data, labels, test_size=model_test_size)

# ---------------------------------------
# Create Datasets, Dataloader, Batch
# ---------------------------------------   
# convert to PyTorch Datasets
Dataset_train = TensorDataset(data_train, labels_train)
Dataset_test  = TensorDataset(data_test, labels_test)

# finally, translate into dataloader objects
DataLoader_train = DataLoader(Dataset_train, shuffle=True, batch_size=batch_size)
DataLoader_test  = DataLoader(Dataset_test, batch_size=Dataset_test.tensors[0].shape[0])

# # check data
# note: observe dataloader by iterate through them
# for X, y in train_loader:
#   print(X.shape, y.shape)
# X, y

# X, y = next(iter(DataLoader_test))
# X, y

#%% Create/Train model

# experiment parameters
l2lambdas = np.linspace(0, .1, 10) # lamda

model_params_0 = [1] 
model_params_1 = l2lambdas
model_params_name = ['Number of Hidden Layer', 'Lambda']

# model parameters
lr = 0.005
epochs = 1000
dr = 0

# store result
list_y_pred_params_0 = []
list_losses_train_params_0 = []
list_acc_ep_train_params_0 = []
list_acc_ep_test_params_0 = []
nParams = np.zeros( (len(model_params_1), len(model_params_0)) )
res_acc = np.zeros( (len(model_params_1), len(model_params_0)) )

# run experiments
for (e, param_0) in enumerate(model_params_0):
    
    # temporary store result
    # list_y_pred_params_1 = []
    # list_losses_train_params_1 = []
    # list_acc_ep_train_params_1 = []
    # list_acc_ep_test_params_1 = []
    
    res_losses_train = np.zeros( (len(model_params_1), epochs) )
    res_acc_ep_train = np.zeros( (len(model_params_1), epochs) )
    res_acc_ep_test = np.zeros( (len(model_params_1), epochs) )
    
    # res_y_pred_train = np.zeros( (batch_size, len(np.unique(labels)), len(model_params_1)) )
    
    # Split training and test set
    # data_train,data_test, labels_train,labels_test = train_test_split(data, labels, test_size=model_test_size)
    
    for (i, param_1) in enumerate(model_params_1):
        
        # Model class instance
        ANNclf_model = ANN_Class(feature_in=4, 
                                 feature_out=3, 
                                 n_hUnit=64, 
                                 n_hLayer=param_0, 
                                 dropout_rate=dr, 
                                 learning_rate=lr,
                                 p_lambda=param_1
                                 )
        
        # result during training per each epoch
        _, model_losses_train, model_acc_ep_train, model_acc_ep_test, _ = ANNclf_model.trainModel(
            DataLoader_train, 
            DataLoader_test, 
            epochs=epochs,
            batch_size=batch_size,
            loss_function='cross-entropy'
            )
        
        res_losses_train[i, :] = model_losses_train.detach().numpy()
        # list_losses_train_params_1.append(res_losses_train)
        
        res_acc_ep_train[i, :] = model_acc_ep_train.detach().numpy()
        # list_acc_ep_train_params_1.append(res_acc_ep_train)
        
        # res_y_pred_train[:, :, i] = model_y_pred_train.detach().numpy()
        # list_y_pred_params_1.append(res_y_pred_train)
        
        # accuracy comparing to test_set per each epoch
        res_acc_ep_test[i, :] = model_acc_ep_test.detach().numpy()
        # list_acc_ep_test_params_1.append(res_acc_ep_test)
        
        # model parameters
        # nParams[i, e] = model_nParams
    
    # Store result
    list_losses_train_params_0.append(res_losses_train)
    list_acc_ep_train_params_0.append(res_acc_ep_train)
    list_acc_ep_test_params_0.append(res_acc_ep_test)
    # list_y_pred_params_0.append(list_y_pred_params_1)


#%% Post-processing results

# --------------------------------------- 
# Only for multiple outputs
# --------------------------------------- 
# Softmax
# sm = nn.Softmax(dim=1)

# average over focus hyperparams and experiments
# mean_y_pred = np.mean(np.mean(np.mean(list_y_pred_params_0, axis=0), axis=0), axis=2)

# create a 1D smoothing filter
def smooth(x, k=5):
    return np.convolve(x, np.ones(k)/k, mode='same')


#%% plot model performance

# plot final losses
fig = plt.figure()
fig.tight_layout()
fig.suptitle(f'Overall Performance', fontweight='bold')

# setting
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)



# plot acc-epoch by model
# plt.figure()
# plt.subplot(2, 2, 1)
# yplot = np.mean(np.array(list_acc_ep_train_params_0[0][0].reshape((epochs, len(DataLoader_train)))), axis=1)
# plt.plot(np.arange(1, epochs+1), yplot, linewidth=1)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy by Epoch')
# plt.legend([f'{i} Hidden Layers'for i in model_params_1])
# plt.show()


# # plot losses-epoch by model
# # plt.figure()
plt.subplot(2, 2, 1)
yplot = np.array(list_losses_train_params_0[0])
plt.plot(np.arange(1, epochs+1), yplot.T, linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses by Epoch')
plt.legend([f'{np.round(i, 2)}'for i in model_params_1])
plt.show()

# plot train-test acc-epoch by model 
# plt.figure()
plt.subplot(2, 2, 2)
y_plot_1 = np.mean(np.array(list_acc_ep_train_params_0[0]), axis=0)
y_plot_2 = np.mean(np.array(list_acc_ep_test_params_0[0]), axis=0)
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.plot(np.arange(1, epochs+1), y_plot_2, linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch')
plt.legend(['Train set', 'Test set'])
plt.show()

# plot train acc-epoch by model 
plt.subplot(2, 2, 3)
y_plot_1 = np.array(list_acc_ep_train_params_0[0]).T
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch [Train set]')
plt.legend([f'{np.round(i, 2)}'for i in model_params_1])
plt.ylim([50,101])
plt.show()

# plot test acc-epoch by model 
plt.subplot(2, 2, 4)
y_plot_1 = np.array(list_acc_ep_test_params_0[0]).T
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch [Test set]')
plt.legend([f'{np.round(i, 2)}'for i in model_params_1])
plt.ylim([50,101])
plt.show()



# # plot train-test acc-epoch by model with convolve
# # plt.figure()
# plt.subplot(2, 2, 4)
# y_plot_1 = np.mean(np.array(list_acc_ep_train_params_0[0]), axis=0)
# y_plot_2 = np.mean(np.array(list_acc_ep_test_params_0[0]), axis=0)
# trim_edge = -2 # idx from end
# plt.plot(np.arange(1, epochs+1)[:trim_edge], smooth(y_plot_1)[:trim_edge], linewidth=1) # remove edge effects last 5 elements
# plt.plot(np.arange(1, epochs+1)[:trim_edge], smooth(y_plot_2)[:trim_edge], linewidth=1) # remove edge effects last 5 elements
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy by Epoch with 1D Convolve')
# plt.legend(['Train set', 'Test set'])
# plt.show()

#%% plot experiment parameters and acc

# average over last 100 epochs
a_p = np.zeros((2, len(model_params_1)))
for i, lis in enumerate(list_acc_ep_train_params_0):
    a_p[0, :] = np.mean(lis[:, -100:], axis=1)
for i, lis in enumerate(list_acc_ep_test_params_0):
    a_p[1, :] = np.mean(lis[:, -100:], axis=1)
    
fig,ax = plt.subplots(1, 2, figsize=(15,5))

ax[0].plot(model_params_1, a_p.T,'o-')
ax[0].set_xlabel('Experiment Parameter')
ax[0].set_ylabel('Average accuracy')
ax[0].legend(['Train','Test'])

ax[1].plot(model_params_1, -np.diff(a_p.T,axis=1),'o-')
ax[1].plot([0, .9], [0, 0],'k--')
ax[1].set_xlabel('Experiment Parameter')
ax[1].set_ylabel('Train-test difference (acc%)')


# all
a_p = np.zeros((2, len(model_params_1)))
for i, lis in enumerate(list_acc_ep_train_params_0):
    a_p[0, :] = np.mean(lis[:, :], axis=1)
for i, lis in enumerate(list_acc_ep_test_params_0):
    a_p[1, :] = np.mean(lis[:, :], axis=1)
   
fig,ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].plot(model_params_1, a_p.T,'o-')
ax[0].set_xlabel('Experiment Parameter')
ax[0].set_ylabel('Average accuracy')
ax[0].legend(['Train','Test'])

ax[1].plot(model_params_1, -np.diff(a_p.T,axis=1),'o-')
ax[1].plot([0, .9], [0, 0],'k--')
ax[1].set_xlabel('Experiment Parameter')
ax[1].set_ylabel('Train-test difference (acc%)')

