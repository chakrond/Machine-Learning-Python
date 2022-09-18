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

# scipy
import scipy.stats as stats

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5 # default 0.8

#%% ANN Class

class ANN_Class(nn.Module):
    
    # Inheritance from parents class
    def __init__(self, 
                 feature_in, 
                 feature_out, 
                 n_hUnit, 
                 n_hLayer, 
                 dropout_rate=0.5, 
                 learning_rate=0.01,
                 w_decay=None,
                 p_lambda=0.01, 
                 ):
        
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
        # weight decay
        self.wDecay = w_decay
        # Lambda
        self.Ld = p_lambda
        
        # ---------------------------------------
        # Number of parameters
        # ---------------------------------------
        # count the total number of weights in the model weight except bias term
        n_weights = 0
        for param_name, weight in self.named_parameters():
            if 'bias' not in param_name:
                n_weights = n_weights + weight.numel()
                
        self.nWeights = n_weights
        
        # ---------------------------------------
        # Optimizer
        # ---------------------------------------
        if w_decay == 'L2':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=p_lambda) # stochastic gradient
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
    
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
                    
                # if L1 is identified
                if self.wDecay == 'L1':
                    # add L1 term
                    L1_term = torch.tensor(0., requires_grad=True)
                    
                    # sum up all abs(weights)
                    for param_name, weight in self.named_parameters():
                        if 'bias' not in param_name:
                            L1_term = L1_term + torch.sum(torch.abs(weight))
    
                    # add to loss term
                    loss = loss + self.Ld*L1_term/self.nWeights
                
                # store loss in every epoch
                losses_batch_train[batch] = loss 
                
                # backpropagation
                self.optimizer.zero_grad()  # set derivative gradient of the model to be zero
                loss.backward()             # back propagation on the computed losses
                self.optimizer.step()       # stochastic gradient
            
                # compute accuracy per batch of training set
                if loss_function == 'cross-entropy':
                    labels_pred_batch_train = torch.argmax(y_pred_train, axis=1) == y_batch_true_train  # pick the highest probability and compare to true labels
                
                if loss_function == 'binary': 
                    labels_pred_batch_train = ((y_pred_train>0.5) == y_batch_true_train).float()  # pick the probability>0.5 and compare to true labels # 100*torch.mean(((predictions>0.5) == labels).float())
                
                # store acc results
                acc_ba_train = 100*torch.sum(labels_pred_batch_train.float()) / len(labels_pred_batch_train)   
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

#%% data characteristic
data = iris.copy()
dc = data.describe()

# # list number of unique values per column
# for i in data.keys():
#   print(f'{i} has {len(np.unique(data[i]))} unique values')
  
# # pairwise plots
# cols2plot = ['fixed acidity', 'volatile acidity', 'citric acid', 'quality']
# sns.pairplot(data[cols2plot], kind='reg', hue='quality')
# plt.show()

# boxplot
fig, ax = plt.subplots(1, figsize=(17,4))
ax = sns.boxplot(data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

# distribution quality values
fig = plt.figure(figsize=(10, 7))

counts = data['species'].value_counts()
plt.bar(list(counts.keys()), counts)
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Species Distribution')
plt.show()

#%% data preprocessing

# copy
data_z = data.copy()

# data features column name
features = data_z.keys().drop('species')

# z-score
for col in features:
    meanval   = np.mean(data_z[col])
    stdev     = np.std(data_z[col], ddof=1)
    data_z[col] = (data_z[col] - meanval) / stdev

# z-score by scipy
# data_z[features] = data_z[features].apply(stats.zscore)

dc_z = data_z.describe()

# boxplot
fig, ax = plt.subplots(1, figsize=(17,4))
ax = sns.boxplot(data=data_z)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

# convert from pandas dataframe to tensor
data_ts = torch.tensor(data_z[features].values).float()

# create a tensor labels [0, 1, 2]
labels_ts = torch.zeros(len(data_z), dtype=torch.long)
# labels_ts[data_z['species'] == 'setosa'] = 0
labels_ts[data_z['species'] == 'versicolor'] = 1
labels_ts[data_z['species'] == 'virginica'] = 2

#%% train/test dataset
model_test_size = 0.2
p_batch_size = 64

# split to find array length
data_train,data_test, labels_train,labels_test = train_test_split(data_ts, labels_ts, test_size=model_test_size)

# ---------------------------------------
# Create Datasets, Dataloader, Batch
# ---------------------------------------   
# convert to PyTorch Datasets
Dataset_train = TensorDataset(data_train, labels_train)
Dataset_test  = TensorDataset(data_test, labels_test)

# finally, translate into dataloader objects
# DataLoader_train = DataLoader(Dataset_train, shuffle=True, batch_size=p_batch_size, drop_last=True) # drop_last=True, drop the last batch if it's not full batch
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
batchsizes          = 2**np.arange(1, 6, 2)

model_params_0      = [1] 
model_params_1      = batchsizes
model_params_name   = ['Number of Hidden Layer', '']

# model parameters
lr = 0.005
epochs = 1000
dr = 0.5
L2lambda = 0.01

# store result
list_y_pred_params_0        = []
list_losses_train_params_0  = []
list_acc_ep_train_params_0  = []
list_acc_ep_test_params_0   = []
# nParams     = np.zeros( (len(model_params_0), len(model_params_1)) )
time_proc   = np.zeros( (len(model_params_0), len(model_params_1)) )

# run experiments
for (e, param_0) in enumerate(model_params_0):
    
    # store result
    res_losses_train = np.zeros( (len(model_params_1), epochs) )
    res_acc_ep_train = np.zeros( (len(model_params_1), epochs) )
    res_acc_ep_test  = np.zeros( (len(model_params_1), epochs) )
    # res_y_pred_train = np.zeros( (batch_size, len(np.unique(labels)), len(model_params_1)) )
    
    
    for (i, param_1) in enumerate(model_params_1):
        
        # timer
        time_start = time.process_time()
        
        
        # Model class instance
        ANNclf_model = ANN_Class(feature_in=features.shape[0], 
                                 feature_out=3, 
                                 n_hUnit=16, 
                                 n_hLayer=param_0, 
                                 dropout_rate=dr, 
                                 learning_rate=lr,
                                 w_decay='L2',
                                 p_lambda=L2lambda
                                 )
        
        
        # batch size experiment***
        # dataLoader
        DataLoader_train = DataLoader(Dataset_train, batch_size=int(param_1), drop_last=True, shuffle=True)
          
        
        # result during training per each epoch
        _, model_losses_train, model_acc_ep_train, model_acc_ep_test, _ = ANNclf_model.trainModel(
            DataLoader_train, 
            DataLoader_test, 
            epochs=epochs,
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
        # nParams[e, i] = model_nParams
        
        # process time
        time_proc[e, i] = time.process_time() - time_start
    
    # Store result
    list_losses_train_params_0.append(res_losses_train)
    list_acc_ep_train_params_0.append(res_acc_ep_train)
    list_acc_ep_test_params_0.append(res_acc_ep_test)
    # list_y_pred_params_0.append(list_y_pred_params_1)

#%% print the model's parameters

params_table = pd.DataFrame()
params_table['Name'] = []
params_table['Weight'] = []
params_table['accumWeight'] = []

n_weights = 0
i = 0
# count weight except bias term
for param_name, weight in ANNclf_model.named_parameters():
    if 'bias' not in param_name:
        n_weights = n_weights + weight.numel()
        params_table.loc[i, 'Name'] = param_name
        params_table.loc[i, 'Weight'] = weight.numel()
        params_table.loc[i, 'accumWeight'] = n_weights
        i+=1

#%% Post-processing results

# --------------------------------------- 
# Only for multiple outputs
# --------------------------------------- 
# Softmax
# sm = nn.Softmax(dim=1)

# average over focus hyperparams and experiments
# mean_y_pred = np.mean(np.mean(np.mean(list_y_pred_params_0, axis=0), axis=0), axis=2)

# create a 1D smoothing filter
def smooth(x, k=10):
    
    res_x = np.zeros(x.shape)
    
    if len(x.shape) == 1:
        res_x = np.convolve(x, np.ones(k)/k, mode='same')
    
    if len(x.shape) == 2:
        for i in range(x.shape[1]):
            res_x[:, i] = np.convolve(x[:, i], np.ones(k)/k, mode='same')
    
    return res_x

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
plt.legend([f'{np.round(i, 4)}'for i in model_params_1])
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
plt.title('Avg Accuracy by Epoch')
plt.legend(['Train set', 'Test set'])
plt.show()

# plot train acc-epoch by model 
plt.subplot(2, 2, 3)
y_plot_1 = np.array(list_acc_ep_train_params_0[0]).T
plt.plot(np.arange(1, epochs+1), smooth(y_plot_1), linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch [Train set]')
plt.legend([f'{np.round(i, 4)}'for i in model_params_1])
plt.ylim([50, 101])
plt.show()

# plot test acc-epoch by model 
plt.subplot(2, 2, 4)
y_plot_1 = np.array(list_acc_ep_test_params_0[0]).T
plt.plot(np.arange(1, epochs+1), smooth(y_plot_1), linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch [Test set]')
plt.legend([f'{np.round(i, 4)}'for i in model_params_1])
plt.ylim([50, 101])
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
# ax[1].plot([0, .9], [0, 0], 'k--')
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
# ax[1].plot([0, .9], [0, 0], 'k--')
ax[1].set_xlabel('Experiment Parameter')
ax[1].set_ylabel('Train-test difference (acc%)')

