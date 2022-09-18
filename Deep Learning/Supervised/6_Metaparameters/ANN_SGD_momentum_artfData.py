# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 01:49:06 2022

@author: Chakron.D
"""

# %% Importing the libraries

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
plt.rcParams['axes.linewidth'] = 0.5  # default 0.8

# %% ANN Class


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
                 p_momentum=None,
                 batch_norm=None,
                 act_lib='torch',
                 activation_fun='relu'  # activation function at hiddne layers
                 ):

        super().__init__()
        # super(ANN_Class, self).__init__()

        # ---------------------------------------
        # Layer structure
        # ---------------------------------------
        self.layers = nn.ModuleDict()  # store layers

        self.n_hLayer = n_hLayer  # number of hidden layers

        self.layers['input'] = nn.Linear(feature_in, n_hUnit)  # input layer

        for i in range(n_hLayer):
            self.layers[f'hidden{i}'] = nn.Linear(
                n_hUnit, n_hUnit)  # hidden layers

            if batch_norm == True:
                self.layers[f'batchNorm{i}'] = nn.BatchNorm1d(n_hUnit)

        self.layers['output'] = nn.Linear(n_hUnit, feature_out)  # output layer

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
        # batch Normalize bool
        self.batchNorm = batch_norm
        # activaiton funciton
        self.actLib = act_lib
        self.actfun = activation_fun

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
            self.optimizer = torch.optim.SGD(self.parameters(
            ), lr=learning_rate, weight_decay=p_lambda, momentum=p_momentum)  # stochastic gradient
        else:
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=learning_rate, momentum=p_momentum)

    # forward pass

    def forward(self, x):

        # input layer
        x = self.layers['input'](x)

        # dropout after input layer
        # training=self.training means to turn off during eval mode
        x = F.dropout(x, p=self.dr, training=self.training)

        if self.actLib == 'torch':
            # activation functions
            actFun = getattr(torch, self.actfun)

        if self.actLib == 'torch.nn':
            # activation functions
            actFun = getattr(torch.nn, self.actfun)

        # hidden layers
        for i in range(self.n_hLayer):

            if self.batchNorm == True:
                x = self.layers[f'batchNorm{i}'](x)

            if self.actLib == 'torch':
                # hidden layer
                # x = F.relu( self.layers[f'hidden{i}'](x) )
                x = actFun(self.layers[f'hidden{i}'](x))

            if self.actLib == 'torch.nn':
                # hidden layer
                x = actFun()(self.layers[f'hidden{i}'](x))

            # dropout
            x = F.dropout(x, p=self.dr, training=self.training)

        # return output layer
        x = self.layers['output'](x)
        return x

    def trainModel(self,
                   DataLoader_train,
                   DataLoader_test,
                   epochs=1000,
                   loss_function=None,
                   comp_acc=None
                   ):

        # ---------------------------------------
        # Metaparams
        # ---------------------------------------
        self.loss_func = loss_function

        # loss function
        if loss_function == 'cross-entropy':
            # already include computing Softmax activation function at the output layer
            loss_fun = nn.CrossEntropyLoss()

        if loss_function == 'binary':
            loss_fun = nn.BCEWithLogitsLoss()  # already combines a Sigmoid layer

        if loss_function == 'MSE':
            loss_fun = nn.MSELoss()  # mean squared error

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
                # predicted result from model
                y_pred_train = self.forward(X_batch_train)

                # compute loss
                # y = target/true value
                loss = loss_fun(y_pred_train, y_batch_true_train)

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

                # store loss in every batch
                losses_batch_train[batch] = loss

                # backpropagation
                self.optimizer.zero_grad()  # set derivative gradient of the model to be zero
                loss.backward()             # back propagation on the computed losses
                self.optimizer.step()       # stochastic gradient

                # compute accuracy per batch of training set
                if loss_function == 'cross-entropy':
                    # pick the highest probability and compare to true labels
                    labels_pred_batch_train = torch.argmax(
                        y_pred_train, axis=1) == y_batch_true_train
                    acc_ba_train = 100 * \
                        torch.sum(labels_pred_batch_train.float()) / \
                        len(labels_pred_batch_train)
                    acc_batch_train[batch] = acc_ba_train

                if loss_function == 'binary':
                    # pick the probability>0.5 and compare to true labels # 100*torch.mean(((predictions>0.5) == labels).float())
                    labels_pred_batch_train = (
                        (y_pred_train > 0.5) == y_batch_true_train).float()
                    acc_ba_train = 100 * \
                        torch.sum(labels_pred_batch_train.float()) / \
                        len(labels_pred_batch_train)
                    acc_batch_train[batch] = acc_ba_train

                # batch increment
                batch += 1
                # ----------------------------------------

            # average batch losses&acc
            losses_ep_train[epoch] = torch.mean(losses_batch_train)
            acc_ep_train[epoch] = torch.mean(acc_batch_train)

            # compute accuracy per epoch of test set
            X_ep_test, y_ep_true_test = next(iter(DataLoader_test))
            _, acc_epoch_test = self.predict(X_ep_test, y_ep_true_test)
            acc_ep_test[epoch] = acc_epoch_test
            # ----------------------------------------

        # total number of trainable parameters in the model
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # ----------------------------------------

        return y_pred_train, losses_ep_train, acc_ep_train, acc_ep_test, n_params

    def predict(self, data, y_true):

        # Make prediction
        self.eval()  # switch training off and no dropout during this mode

        # Model Accuracy
        if self.loss_func == 'cross-entropy':
            with torch.no_grad():  # deactivates autograd
                predictions = self.forward(data)
            labels_pred = torch.argmax(predictions, axis=1) == y_true
            total_acc = 100*torch.sum(labels_pred.float()) / len(labels_pred)

        if self.loss_func == 'binary':
            with torch.no_grad():  # deactivates autograd
                predictions = self.forward(data)
            labels_pred = ((predictions > 0.5) == y_true).float()
            total_acc = 100*torch.sum(labels_pred.float()) / len(labels_pred)

        if self.loss_func == 'MSE':
            loss_fun = nn.MSELoss()
            with torch.no_grad():  # deactivates autograd
                predictions = self.forward(data)
            total_acc = loss_fun(predictions, y_true)

        return predictions, total_acc

# %% Import dataset

# create data


nPerClust = 300
blur = 1

A = [1, 1]
B = [5, 1]
C = [4, 4]

# generate data
a = [A[0]+np.random.randn(nPerClust)*blur, A[1] +
     np.random.randn(nPerClust)*blur]
b = [B[0]+np.random.randn(nPerClust)*blur, B[1] +
     np.random.randn(nPerClust)*blur]
c = [C[0]+np.random.randn(nPerClust)*blur, C[1] +
     np.random.randn(nPerClust)*blur]

# true labels
labels_np = np.hstack((np.zeros((nPerClust)),
                       np.ones((nPerClust)),
                       1+np.ones((nPerClust))))

# concatanate into a matrix
data_np = np.hstack((a, b, c)).T

# show the data
fig = plt.figure(figsize=(5, 5))
plt.plot(data_np[np.where(labels_np == 0)[0], 0],
         data_np[np.where(labels_np == 0)[0], 1], 'bs', alpha=.5)
plt.plot(data_np[np.where(labels_np == 1)[0], 0],
         data_np[np.where(labels_np == 1)[0], 1], 'ko', alpha=.5)
plt.plot(data_np[np.where(labels_np == 2)[0], 0],
         data_np[np.where(labels_np == 2)[0], 1], 'r^', alpha=.5)
plt.title('The qwerties!')
plt.xlabel('qwerty dimension 1')
plt.ylabel('qwerty dimension 2')
plt.show()

# %% data characteristic
# dc = data.describe()

# # list number of unique values per column
# for i in data.keys():
#   print(f'{i} has {len(np.unique(data[i]))} unique values')

# # pairwise plots
# cols2plot = ['fixed acidity', 'volatile acidity', 'citric acid', 'quality']
# sns.pairplot(data[cols2plot], kind='reg', hue='quality')
# plt.show()

# boxplot
fig, ax = plt.subplots(1, figsize=(17, 4))
ax = sns.boxplot(data=data_np)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_title('Data values Distribution')
plt.show()


# # inspect values
# value_name = 'residual sugar'

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.suptitle(f'{value_name} Distribution', fontweight='bold')

# plt.plot(data[value_name].values,'s')
# plt.xlabel('Data observation')
# plt.ylabel(f'{value_name}')
# plt.show()

# # distribution the values
# plt.subplot(2, 1, 2)
# counts = data[value_name].value_counts()
# plt.bar(list(counts.keys()), counts)
# plt.xlabel(f'{value_name} value')
# plt.ylabel('Count')
# plt.show()

# %% data preprocessing

# z-score all variables except for quality label

# copy
# data_z = data.copy()
data_z = data_np

# data features column name
# features = data_z.keys() #.drop('quality')

# z-score
# for col in features:
#     meanval   = np.mean(data_z[col])
#     stdev     = np.std(data_z[col], ddof=1)
#     data_z[col] = (data_z[col] - meanval) / stdev

# z-score by scipy
data_z = stats.zscore(data_z)

# dc_z = data_z.describe()

# boxplot
fig, ax = plt.subplots(1, figsize=(17, 4))
ax = sns.boxplot(data=data_z)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()

# # inspect values
# plt.figure()
# plt.plot(data_z[value_name].values,'s')
# plt.xlabel('Data observation')
# plt.ylabel(f'{value_name} (normalized)')
# plt.show()

# create a new column for binarized boolean quality [0 or 1]
# data_z['boolQuality'] = 0
# data_z['boolQuality'][data_z['quality']>5] = 1 # wine quality > 5 will assigned as 1
# data_z[['quality', 'boolQuality']]

# # Inspect the correlation matrix
# fig = plt.figure(figsize=(8,8))

# corrcoef_mat = np.corrcoef(data_z.T)
# plt.imshow(corrcoef_mat, vmin=-0.3, vmax=0.3)
# plt.xticks(range(len(data_z.keys())), labels=data_z.keys(), rotation=90)
# plt.yticks(range(len(data_z.keys())), labels=data_z.keys())
# plt.colorbar()
# plt.title('Data correlation matrix')
# plt.show()

# %% convert to tensor

data_ts = torch.tensor(data_np).float()
y_ts = torch.tensor(labels_np).long()
# y_ts = y_ts.reshape((-1, 1))

# %% train/test dataset
model_test_size = 0.2
p_batch_size = 32

# split to find array length
data_train, data_test, y_train, y_test = train_test_split(
    data_ts, y_ts, test_size=model_test_size)

# ---------------------------------------
# Create Datasets, Dataloader, Batch
# ---------------------------------------
# convert to PyTorch Datasets
Dataset_train = TensorDataset(data_train, y_train)
Dataset_test = TensorDataset(data_test, y_test)

# finally, translate into dataloader objects
# drop_last=True, drop the last batch if it's not full batch
DataLoader_train = DataLoader(
    Dataset_train, shuffle=True, batch_size=p_batch_size, drop_last=True)
DataLoader_test = DataLoader(
    Dataset_test, batch_size=Dataset_test.tensors[0].shape[0])

# # check data
# note: observe dataloader by iterate through them
# for X, y in train_loader:
#   print(X.shape, y.shape)
# X, y

# X, y = next(iter(DataLoader_test))
# X, y

# %% Create/Train model

# experiment parameters
momenta = [0, .5, .9, .95, .999]

model_params_0 = [2]
model_params_1 = momenta
# model_params_name   = ['Number of Hidden Layer', '']

# model parameters
lr = 0.01
epochs = 1000
dr = 0.5
L2lambda = 0.01

# store result
list_y_pred_params_0 = []
list_losses_train_params_0 = []
list_acc_ep_train_params_0 = []
list_acc_ep_test_params_0 = []
# nParams     = np.zeros( (len(model_params_0), len(model_params_1)) )
time_proc = np.zeros((len(model_params_0), len(model_params_1)))

# run experiments
for (e, param_0) in enumerate(model_params_0):

    # store result
    res_losses_train = np.zeros((len(model_params_1), epochs))
    res_acc_ep_train = np.zeros((len(model_params_1), epochs))
    res_acc_ep_test = np.zeros((len(model_params_1), epochs))
    # res_y_pred_train = np.zeros( (batch_size, len(np.unique(labels)), len(model_params_1)) )

    for (i, param_1) in enumerate(model_params_1):

        # timer
        time_start = time.process_time()

        # Model class instance
        ANNclf_model = ANN_Class(feature_in=data_train.shape[1],
                                 feature_out=3,
                                 n_hUnit=32,
                                 n_hLayer=param_0,
                                 dropout_rate=dr,
                                 learning_rate=lr,
                                 w_decay='L2',
                                 p_lambda=L2lambda,
                                 p_momentum=param_1,
                                 batch_norm=None,
                                 act_lib='torch',
                                 activation_fun='relu'
                                 )

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

# %% print the model's parameters

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
        i += 1

# %% Post-processing results

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

# %% plot model performance


# plot
fig = plt.figure()
fig.tight_layout()
fig.suptitle(f'Overall Performance', fontweight='bold')

# setting
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=None, hspace=0.25)


# plot train-test acc-epoch by model
# plt.figure()
plt.subplot(1, 2, 1)
y_plot_1 = np.array(list_acc_ep_train_params_0[0]).T
y_plot_2 = np.mean(np.array(list_acc_ep_train_params_0[0]), axis=0)
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.plot(np.arange(1, epochs+1), y_plot_2, 'k', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch(Train set)')
plt.legend([f'momentum {param}' for param in model_params_1] + ['Mean'])

plt.subplot(1, 2, 2)
y_plot_1 = np.array(list_acc_ep_test_params_0[0]).T
y_plot_2 = np.mean(np.array(list_acc_ep_test_params_0[0]), axis=0)
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.plot(np.arange(1, epochs+1), y_plot_2, 'k', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch(test set)')
plt.legend([f'momentum {param}' for param in model_params_1] + ['Mean'])

#%%

# plot train-test acc-epoch by model
plt.figure()
# plt.subplot(1, 2, 1)
y_plot_1 = np.mean(np.array(list_acc_ep_train_params_0[0]), axis=0)
y_plot_2 = np.mean(np.array(list_acc_ep_test_params_0[0]), axis=0)
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.plot(np.arange(1, epochs+1), y_plot_2, linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Avg Accuracy by Epoch')
plt.legend(['Train set', 'Test set'])

#%%
# get the categorical predictions ***from the last model
yHat, _ = ANNclf_model.predict(data_ts, y_ts)
predictions = torch.argmax(yHat, axis=1)

# and plot those against the real data
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(predictions, 'o', label='Predicted values')
plt.plot(y_ts+.2, 's', label='True values')
plt.xlabel('Stimulus number')
plt.ylabel('Category')
plt.yticks([0, 1, 2])
plt.ylim([-1, 3])
plt.legend()

# accuracy by category
accuracy = (predictions == y_ts).float()

# compute overall accuracy
totalAcc = torch.mean(100*accuracy).item()

# and average by group
accuracyByGroup = np.zeros(3)
for i in [0, 1, 2]:
    accuracyByGroup[i] = 100*torch.mean(accuracy[y_ts == i])

plt.subplot(1, 2, 2)
plt.bar(range(3), accuracyByGroup)
plt.ylim([80, 100])
plt.xticks([0, 1, 2])
plt.xlabel('Group')
plt.ylabel('Accuracy (%)')
plt.title(f'Final accuracy = {totalAcc:.2f}%')


sm = nn.Softmax(dim=1)
fig = plt.figure()
fig.suptitle(f'Overall performance of model prediction', fontweight='bold')

plt.subplot(1, 2, 1)
plt.plot(sm(yHat), 's-', markerfacecolor='w', alpha=0.8)
plt.xlabel('Stimulus number')
plt.ylabel('Probability (Softmax)')
plt.legend([0, 1, 2])
plt.show()

plt.subplot(1, 2, 2)
plt.plot(yHat, 'o-', markerfacecolor='w', alpha=0.8)
plt.xlabel('Stimulus number')
plt.ylabel('Probability (raw)')
plt.legend([0, 1, 2])
plt.show()


# %% plot experiment parameters

# average over last 100 epochs
a_p = np.zeros((2, len(model_params_1)))
for i, lis in enumerate(list_acc_ep_train_params_0):
    a_p[0, :] = np.mean(lis[:, -100:], axis=1)
for i, lis in enumerate(list_acc_ep_test_params_0):
    a_p[1, :] = np.mean(lis[:, -100:], axis=1)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))


ax[0].plot(model_params_1, a_p.T, 'o-')
ax[0].set_xlabel('Experiment Parameter')
ax[0].set_ylabel('Average accuracy')
ax[0].legend(['Train', 'Test'])

ax[1].plot(model_params_1, -np.diff(a_p.T, axis=1), 'o-')
# ax[1].plot([0, .9], [0, 0], 'k--')
ax[1].set_xlabel('Experiment Parameter last 100 epochs')
ax[1].set_ylabel('Train-test difference (acc%)')


# all
a_p = np.zeros((2, len(model_params_1)))
for i, lis in enumerate(list_acc_ep_train_params_0):
    a_p[0, :] = np.mean(lis[:, :], axis=1)
for i, lis in enumerate(list_acc_ep_test_params_0):
    a_p[1, :] = np.mean(lis[:, :], axis=1)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(model_params_1, a_p.T, 'o-')
ax[0].set_xlabel('Experiment Parameter')
ax[0].set_ylabel('Average accuracy')
ax[0].legend(['Train', 'Test'])

ax[1].plot(model_params_1, -np.diff(a_p.T, axis=1), 'o-')
# ax[1].plot([0, .9], [0, 0], 'k--')
ax[1].set_xlabel('Experiment Parameter')
ax[1].set_ylabel('Train-test difference (acc%)')
