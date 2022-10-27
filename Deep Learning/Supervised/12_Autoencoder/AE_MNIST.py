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
import sys
import copy

# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter

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
import sklearn.metrics as skm

# scipy
import scipy.stats as stats

# set directory
import inspect, os.path
filename = inspect.getframeinfo(inspect.currentframe()).filename
path_dir     = os.path.dirname(os.path.abspath(filename))
os.chdir(path_dir) # set the working directory
cd = os.getcwd()

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5  # default 0.8

#%% Import FFN Class

sys.path.append('../')
import Class.AE as AE

#%% Functions

# Data split & data loader function
def dSplit(data_ts,
           y_ts,
           model_test_size=0.2,
           p_batch_size=32,
           ):
    
    # split to find array length
    data_train, data_test, y_train, y_test = train_test_split(
        data_ts, y_ts, test_size=model_test_size)

    # ---------------------------------------
    # Create Datasets, Dataloader, Batch
    # ---------------------------------------
    # convert to PyTorch Datasets
    Dataset_train = TensorDataset(data_train, y_train)
    Dataset_test = TensorDataset(data_test, y_test)
    size = [Dataset_train.tensors[0].shape, Dataset_test.tensors[0].shape]

    # finally, translate into dataloader objects
    # drop_last=True, drop the last batch if it's not full batch
    DataLoader_train = DataLoader(Dataset_train, shuffle=True, batch_size=p_batch_size, drop_last=True)
    DataLoader_test = DataLoader(Dataset_test, batch_size=Dataset_test.tensors[0].shape[0])
    
    return DataLoader_train, DataLoader_test, size


# Post-processing results

# Softmax
# sm = nn.Softmax(dim=1)

# average over focus hyperparams and experiments
# mean_y_pred = np.mean(np.mean(np.mean(list_y_pred_params_0, axis=0), axis=0), axis=2)

# create a 1D smoothing filter

# ---------------------------------------
# Only for multiple outputs
# ---------------------------------------
def smooth(x, k=10):

    res_x = np.zeros(x.shape)

    if len(x.shape) == 1:
        res_x = np.convolve(x, np.ones(k)/k, mode='same')

    if len(x.shape) == 2:
        for i in range(x.shape[1]):
            res_x[:, i] = np.convolve(x[:, i], np.ones(k)/k, mode='same')

    return res_x


# noise augmentation
def noise(data, divd):
    
    data = data / np.max(data)
    data_noise = data + np.random.random_sample(data.shape) / divd
    data_noise = data_noise / np.max(data_noise)

    return data_noise

# scikitlearn model score
def skmScore(true, prediction):
    
    data_metric = {
        'Accuracy': 0,
        'Precision': 0,
        'Recall': 0,
        'F1': 0,
        }
    
    data_metric['Accuracy'] = skm.accuracy_score(true, prediction)
    data_metric['Precision'] = skm.precision_score(true, prediction)
    data_metric['Recall'] = skm.recall_score(true, prediction)
    data_metric['F1'] = skm.f1_score(true, prediction)    
    
    return data_metric
 

#%% Import dataset

# MNIST data
fname = 'z:\python traning\my codes\machine learning\deep learning\supervised\99_Datasets\MNIST\mnist_train_small.csv'
fdata = np.loadtxt(open(fname, 'rb'), delimiter=',')

# extract data and labels
labels = fdata[:, 0]
data = fdata[:, 1:]

#%% Data inspection

# show a few random  of 28x28 img
fig, axs = plt.subplots(3, 4, figsize=(10, 6))

# generate random number
randimg2show = np.random.randint(data.shape[0], size=len(axs.flatten()))


for i, ax in enumerate(axs.flatten()):
  
    # create the image (must be reshaped!)
    img = np.reshape(data[randimg2show[i],:], (28, 28))
    ax.imshow(img, cmap='gray')
    
    # title
    ax.set_title('The number %i'%labels[randimg2show[i]])

plt.suptitle('28x28 Image Data', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, .95])
plt.show()


# show a few random digits of data vector
fig,axs = plt.subplots(3, 4, figsize=(10, 6))

for i, ax in enumerate(axs.flatten()):

    # create the image
    ax.plot(data[randimg2show[i],:],'ko')
    
    # title
    ax.set_title('The number %i'%labels[randimg2show[i]])

plt.suptitle('How the model sees the data(vector)', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, .95])
plt.show()

#%% Correlation

# example 7s

# find indices of all the 7's in the dataset
idx7 = np.where(labels==7)[0]

# draw the first 12
fig,axs = plt.subplots(2, 6, figsize=(15, 6))

for i,ax in enumerate(axs.flatten()):
    img = np.reshape(data[idx7[i], :], (28, 28))
    ax.imshow(img,cmap='gray')
    ax.axis('off')

plt.suptitle("Example 7's",fontsize=20)
plt.tight_layout(rect=[0, 0, 1, .95])

# let's see how they relate to each other by computing spatial correlations
C = np.corrcoef(data[idx7,:])

# and visualize
fig,ax = plt.subplots(1, 3, figsize=(16, 6))
ax[0].imshow(C, vmin=0, vmax=1)
ax[0].set_title("Correlation across all 7's")

# extract the unique correlations and show as a scatterplot
uniqueCs = np.triu(C, k=1)
uniqueCsF = uniqueCs.flatten()
ax[1].hist(uniqueCsF[uniqueCsF!=0], bins=100)
ax[1].set_title('All unique correlations')
ax[1].set_xlabel("Correlations of 7's")
ax[1].set_ylabel('Count')

# show all 7's together
aveAll7s = np.reshape( np.mean(data[idx7, :], axis=0), (28, 28))
ax[2].imshow(img, cmap='gray')
ax[2].set_title("All 7's averaged together")

plt.tight_layout()
plt.show()

#%% data characteristic


#%% data preprocessing

# normalize the data to [0 1]
dataNorm = data / np.max(data)

fig,ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(data.flatten(), 50)
ax[0].set_xlabel('Pixel intensity values')
ax[0].set_ylabel('Count')
ax[0].set_title('Histogram of original data')
ax[0].set_yscale('log')

ax[1].hist(dataNorm.flatten(), 50)
ax[1].set_xlabel('Pixel intensity values')
ax[1].set_ylabel('Count')
ax[1].set_title('Histogram of normalized data')
ax[1].set_yscale('log')

plt.show()

#%% convert to tensor

# independent test set
# convert to tensor
data_train_ts = torch.tensor(dataNorm).float()
# y_test_ind_ts = torch.tensor(y_test_ind).long() # .long() (integer and for cross-entropy, multi class)
# y_test_ind_ts = y_test_ind_ts.reshape((-1, 1)).long() # .float() and reshape only for binary
# y_test_ind_ts = torch.tensor(y_test_ind).reshape((-1, 1)).float()

# convert to dataset and dataLoader
b_size = 32
Dataset_train = TensorDataset(data_train_ts, data_train_ts)
DataLoader_train = DataLoader(Dataset_train, batch_size=b_size)

# %% Create/Train model

# experiment parameters
nenUnit = [128]
# dataLoaderSet = [[DataLoader_train, DataLoader_dev], [DataLoader_train_na, DataLoader_dev_na]]
nlatUnit = [50]
 
model_params_0 = nenUnit
model_params_1 = nlatUnit

# model parameters
lr = 0.001
epochs = 10
dr = 0.5
L2lambda = 0.01

# store result
# list_y_pred_params_0 = []
list_losses_train_params_0 = []
list_acc_ep_train_params_0 = []
list_acc_ep_test_params_0 = []
# list_w_histx_params_0 = []
list_w_histy_params_0 = []
# nParams     = np.zeros( (len(model_params_0), len(model_params_1)) )
time_proc = np.zeros((len(model_params_0), len(model_params_1)))

# store predicted data from trained model
list_y_pred_model_params_0 = []



# store the best trained model
bestModel = {'Accuracy':0,
             'net':None,
             'params': (None, None),
             }

# run experiments
for (e, param_0) in enumerate(model_params_0):

    # store result
    res_losses_train = np.zeros((len(model_params_1), epochs))
    res_acc_ep_train = np.zeros((len(model_params_1), epochs))
    res_acc_ep_test = np.zeros((len(model_params_1), epochs))
    res_w_histx = np.zeros((epochs, 100, len(model_params_1)))
    res_w_histy = np.zeros((epochs, 100, len(model_params_1)))
    # res_y_pred_train = np.zeros( (batch_size, len(np.unique(labels)), len(model_params_1)) )
    
    # store predicted data from trained model
    y_pred_model = []

    for (i, param_1) in enumerate(model_params_1):

        # timer
        time_start = time.process_time()

        # Model class instance
        AE_model = AE.AE_Class()
        AE_model.setParams(feature_in=dataNorm.shape[1],
                                 feature_out=dataNorm.shape[1],
                                 n_enUnit=param_0,
                                 n_latUnit=param_1,
                                 learning_rate=lr,
                                 activation_fun='relu',
                                 activation_fun_out='sigmoid',
                                 optim_fun='Adam',
                                 w_fro='weight'
                                 )

        # # *************************************************************
        # # weight initialize
        # # *************************************************************
        # if i > 10:
        #     # weight initialize
        #     met = getattr(nn.init, 'xavier_normal_')
        #     FFN_model.named_parameters()
        #     # change the weights (leave biases as Kaiming [default])
        #     for p in FFN_model.named_parameters():
        #       if 'weight' in p[0]:
        #         met(p[1].data)
        #         # thesedata = p[1].data.numpy().flatten()


        # dict
        # dataloader_dict = {
        #     'DataLoader_train': param_1[0],
        #     'DataLoader_test': param_1[1],
        #     }

        dataloader_dict = {
            'DataLoader_train': DataLoader_train,
            'DataLoader_test': DataLoader_train,
            }

        # result during training per each epoch
        _, model_losses_train, model_acc_ep_train, model_acc_ep_test, _, model_w_histx, model_w_histy = AE_model.trainModel(
            **dataloader_dict,
            epochs=epochs,
            loss_function='MSE',
            comp_acc_test=True,
            comp_w_hist=True,
            comp_w_change=True
        )

        res_losses_train[i, :] = model_losses_train.detach().numpy()

        res_acc_ep_train[i, :] = model_acc_ep_train.detach().numpy()

        # res_y_pred_train[:, :, i] = model_y_pred_train.detach().numpy()

        # accuracy comparing to test_set per each epoch
        res_acc_ep_test[i, :] = model_acc_ep_test.detach().numpy()

        # histogram weight
        # res_w_histx[:, :, i] = model_w_histx
        res_w_histy[:, :, i] = model_w_histy

        # model parameters
        # nParams[e, i] = model_nParams

        # process time
        time_proc[e, i] = time.process_time() - time_start
        
        # predict data from trained model
        # y_pred, _ = FFN_model.predict(data_ts, y_ts)
        # y_pred_model.append(y_pred.detach().numpy())
        
        
    # Store result
    list_losses_train_params_0.append(res_losses_train)
    list_acc_ep_train_params_0.append(res_acc_ep_train)
    list_acc_ep_test_params_0.append(res_acc_ep_test)
    # list_w_histx_params_0.append(res_w_histx)
    list_w_histy_params_0.append(res_w_histy)
    # list_y_pred_params_0.append(res_y_pred_train)
    # list_y_pred_model_params_0.append(y_pred_model)
    
    # Store best model
    mean_acc = np.mean(list_acc_ep_test_params_0[-1][-1, :-5])
    if mean_acc > bestModel['Accuracy']:
        bestModel['Accuracy'] = mean_acc
        bestModel['net'] = copy.deepcopy(AE_model.state_dict())
        bestModel['params'] = (e, i)

#%% explore the model's parameters

params_table = pd.DataFrame()
params_table['Name'] = []
params_table['Weight'] = []
params_table['accumWeight'] = []

# get weight data
layer_weight = {}

n_weights = 0
i = 0
# count weight except bias term
for param_name, weight in AE_model.named_parameters():
    if 'weight' in param_name:
        n_weights = n_weights + weight.numel()
        params_table.loc[i, 'Name'] = param_name
        params_table.loc[i, 'Weight'] = weight.numel()
        params_table.loc[i, 'accumWeight'] = n_weights
        params_table.loc[i, 'grad'] = weight.requires_grad
        layer_weight[f'{param_name}'] = weight.data
        i += 1

print(params_table)

#%% Weight changes
WConds = AE_model.trainWConds
WChange = AE_model.trainWChange

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# plot Condition number of weight matrix
ax[0].plot(WConds.T)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('')
ax[0].set_title('Condition number of weight matrix')

# plot Weight Change of weight matrix (Frobenius norm)
ax[1].plot(WChange.T)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('')
ax[1].set_title('Weight Change(Norm Frobenius)')

#%% explore the model's layer

net_input_layer = vars(AE_model.layers['input'])
net_input_layer_weight = AE_model.layers['input'].weight

# plot histogram of weight
plt.figure()
plt.hist(net_input_layer_weight.detach().flatten(), 40)
plt.xlabel('Weight value')
plt.ylabel('Count')
plt.title('Distribution of initialized weight at input layer')


# weight at all layers
W = np.array([])

# get set of weights from each layer
for layer in AE_model.layers:
    W = np.concatenate((W, AE_model.layers[f'{layer}'].weight.detach().flatten().numpy() ))

# compute histogram
histy, histx = np.histogram(W, bins=np.linspace(-.8, .8, 101), density=True)
histx = (histx[1:] + histx[:-1])/2 # correct the dimension

# plot
fig = plt.figure()
fig.tight_layout()
fig.suptitle(f'Weight Histogram', fontweight='bold')

# setting
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=None, hspace=0.25)

plt.subplot(1, 2, 1)
# plot histogram of weight (bar)
plt.hist(W, bins=100)
plt.xlabel('Weight value')
plt.ylabel('Count')
plt.title('Distribution of weights at all layers')

plt.subplot(1, 2, 2)
plt.plot(histx, histy)
plt.ylim([0, np.max(histy)*1.1])
plt.title('Histograms of weights')
plt.xlabel('Weight value')
plt.ylabel('Density')

#%% Model's weight histogram of each epoch

# show the histogram of the weights

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

model_w_histy = list_w_histy_params_0[0][:, :, -1]

# plot histogram of weight (bar)
ax[0].hist(W, bins=100)
ax[0].set_xlabel('Weight value')
ax[0].set_ylabel('Count')
ax[0].set_title('Distribution of weights at all layers')


# plot histogram of weight (line)
w_len = len(model_w_histy)
for i in range(model_w_histy.shape[0]-1):
    ax[1].plot(model_w_histx, model_w_histy[i, :], color=[1-i/w_len, 0.3, i/w_len], alpha=0.5)
ax[1].plot(model_w_histx, model_w_histy[-1, :], color=[1-model_w_histy.shape[0]/w_len, 0.3, model_w_histy.shape[0]/w_len], label='Final Weights')
ax[1].set_title('Histograms of weights')
ax[1].set_xlabel('Weight value')
ax[1].set_ylabel('Density')
ax[1].legend()
ax[1].set_ylim([0, np.max(model_w_histy)*1.1])

# plot image of weight with epoch
ax[2].imshow(model_w_histy, vmin=0, vmax=0.8,
             extent=[model_w_histx[0], model_w_histx[-1], 0, 99], aspect='auto', origin='upper', cmap='turbo')
ax[2].set_xlabel('Weight value')
ax[2].set_ylabel('Training epoch')
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
ax[2].set_title('Image of weight histograms')

plt.show()


#%% plot model performance


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
y_plot_1 = np.array(list_losses_train_params_0[0]).T
y_plot_2 = np.mean(np.array(list_losses_train_params_0[0]), axis=0)
t, p = stats.ttest_ind(y_plot_1, y_plot_2)
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.plot(np.arange(1, epochs+1), y_plot_2, 'k', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss by Epoch(Train set)')
# plt.legend([f'{param}' for param in model_params_1] + ['Mean'])
# plt.legend(['Kaiming', 'Xavier'])

plt.subplot(1, 2, 2)
y_plot_1 = np.array(list_acc_ep_test_params_0[0]).T
y_plot_2 = np.mean(np.array(list_acc_ep_test_params_0[0]), axis=0)
# plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
t, p = stats.ttest_ind(y_plot_1, y_plot_2)
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.plot(np.arange(1, epochs+1), y_plot_2, 'k', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss by Epoch(test set)')
# plt.legend([f'{param}' for param in model_params_1] + ['Mean'])
# plt.legend(['Kaiming', 'Xavier'])

#%% plot train-test acc-epoch by model
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

#%% Prediction from trained model

# sample to show
sample2show = np.random.randint(0, len(dataNorm))

# run the model through for the test data
X = torch.tensor(dataNorm[sample2show, :]).float()
predictions, _ = AE_model.predict(X, X)
predicitons = torch.sigmoid(predictions)

plt.subplot(1, 2, 1)
plt.imshow(np.reshape(dataNorm[sample2show, :], (28, 28)), cmap='gray')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(np.reshape(predicitons, (28, 28)), cmap='gray')
plt.title('Prediction')
# plt.xlabel('Number')
# plt.ylabel('Evidence for that number')
# plt.title(f'True number was {y[sample2show]}')
plt.show()

#%% Prediction from noised data

# plot noise
plt.imshow(torch.rand_like(data_train_ts[0, :]).view(28, 28).detach(), cmap='gray')

# add noise
X_data_noise = data_train_ts + torch.rand_like(data_train_ts[0, :])/4

# clip at 1
X_data_noise[X_data_noise>1] = 1

# random idx
randId = np.random.choice(len(data_train_ts), 5, replace=False)

# run the model through for the test data
X = torch.tensor(X_data_noise[randId, :]).float()
predictions, _ = AE_model.predict(X, X)
predicitons = torch.sigmoid(predictions)


# plot all
fig = plt.figure(constrained_layout=True)
fig.suptitle('MNIST AutoEncoder')

# create 3x1 subfigs
subfigs = fig.subfigures(nrows=3, ncols=1)
subfigs[0].suptitle('Original')
subfigs[1].suptitle('Added noise')
subfigs[2].suptitle('AutoEncoder Denoise')

# create 1x5 subplots per subfig
axs0 = subfigs[0].subplots(nrows=1, ncols=5)
axs1 = subfigs[1].subplots(nrows=1, ncols=5)
axs2 = subfigs[2].subplots(nrows=1, ncols=5)

for i, idx in enumerate(randId):
    axs0[i].imshow(data_train_ts[idx, :].view(28, 28).detach(), cmap='gray')
    axs1[i].imshow(X_data_noise[idx, :].view(28, 28).detach(), cmap='gray')
    axs2[i].imshow(predicitons[i].view(28, 28).detach(), cmap='gray')
    
    for z in range(3):
        eval(f'axs{z}[i].set_xticks([]), axs{z}[i].set_yticks([])')
        
plt.show()

