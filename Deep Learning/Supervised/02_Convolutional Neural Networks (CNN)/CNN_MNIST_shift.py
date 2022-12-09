# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 21:38:25 2022

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
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
import torchvision.transforms as T
from torchsummary import summary

# Seaborn
import seaborn as sns

# sckitlearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
import sklearn.metrics as skm

# scipy
import scipy.stats as stats
from scipy.signal import convolve2d

# read image
from imageio import imread

# set directory
import inspect, os.path
filename = inspect.getframeinfo(inspect.currentframe()).filename
path_dir = os.path.dirname(os.path.abspath(filename))
os.chdir(path_dir) # set the working directory
cd = os.getcwd()

# SSL
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5  # default 0.8
plt.rcParams["figure.subplot.hspace"] = 0.25

#%% Import Class & Functions

# CNN Class
sys.path.append('../')
import Class.CNN as CNN

# Function
from Function.dSplit import dSplit

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

plt.suptitle('How the FFN model sees the data(vector)', fontsize=20)
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

# reshape to 2D
dataNorm = dataNorm.reshape(dataNorm.shape[0], 1, 28, 28)

#%% divide independent set
data_ss, data_test_ind, y_ss, y_test_ind = train_test_split(dataNorm, labels, test_size=0.01)


# independent test set
# convert to tensor
data_test_ind_ts = torch.tensor(data_test_ind).float()
y_test_ind_ts = torch.tensor(y_test_ind).long() # .long() (integer and for cross-entropy, multi class)
# y_test_ind_ts = y_test_ind_ts.reshape((-1, 1)).long() # .float() and reshape only for binary
# y_test_ind_ts = torch.tensor(y_test_ind).reshape((-1, 1)).float()

# convert to dataset and dataLoader
Dataset_test = TensorDataset(data_test_ind_ts, y_test_ind_ts)
DataLoader_test = DataLoader(Dataset_test, batch_size=Dataset_test.tensors[0].shape[0])

# %% train/dev dataset

# train set and dev set
# convert to tensor
data_ts = torch.tensor(data_ss).float()
y_ts = torch.tensor(y_ss).long() # .long() (integer and for cross-entropy, multi class)
# y_ts = y_ts.reshape((-1, 1)).long() # .float() and reshape only for binary
# y_ts = torch.tensor(y_ss).reshape((-1, 1)).float() # MSE (regression)

# params
dev_size = 0.2
b_size = 32

# train and dev set
DataLoader_train, DataLoader_dev, size = dSplit(data_ts, y_ts, model_test_size=dev_size, p_batch_size=b_size)


# # noise augmented set
# DataLoader_train_na, DataLoader_dev_na, _ = dSplit(data_ts_na, y_ts_na, model_test_size=test_size, p_batch_size=b_size)


# data size summary
d = {'Name': ['Total data', 'Training data', 'Devset data', 'Test data'],
     'Size': [data.shape, size[0], size[1], data_test_ind_ts.shape]}

size_table = pd.DataFrame(data=d)
print(size_table)


# check data
# note: observe dataloader by iterate through them
# for X, y in DataLoader_train:
#   print(X.shape, y.shape)
# X, y

# X, y = next(iter(DataLoader_train))
# X, y

#%% shift the images

# train set
for i in range(DataLoader_train.dataset.tensors[0].shape[0]):
  
    # get the image
    img = DataLoader_train.dataset.tensors[0][i, 0, :, :]
    
    # reshape and roll by max. 10 pixels
    randroll = np.random.randint(-10, 11)
    img = torch.roll(img, randroll, dims=1)
    
    # put back
    DataLoader_train.dataset.tensors[0][i, 0, :, :] = img


# dev set
for i in range(DataLoader_dev.dataset.tensors[0].shape[0]):
  
    # get the image
    img = DataLoader_dev.dataset.tensors[0][i, 0, :, :]
    
    # reshape and roll by max. 10 pixels
    randroll = np.random.randint(-10, 11)
    img = torch.roll(img, randroll, dims=1)
    
    # put back
    DataLoader_dev.dataset.tensors[0][i, 0, :, :] = img
    
    
# show example
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

randidx = np.random.randint(0, DataLoader_train.dataset.tensors[0].shape[0])
ax[0].imshow(DataLoader_train.dataset.tensors[0][randidx, 0, :, :])
ax[0].set_title('Train set')

randidx = np.random.randint(0, DataLoader_dev.dataset.tensors[0].shape[0])
ax[1].imshow(DataLoader_dev.dataset.tensors[0][randidx, 0, :, :])
ax[1].set_title('Dev set')


# %% Create/Train model

# experiment parameters
optimTypes = ['Adam']
# dataLoaderSet = [[DataLoader_train, DataLoader_dev], [DataLoader_train_na, DataLoader_dev_na]]
nUnits = [50]

model_params_0 = optimTypes
model_params_1 = nUnits
# model_params_name   = ['Number of Hidden Layer', '']

# model parameters
lr = 0.001
epochs = 10
dr = 0

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

# Layers inputs
convLayers = {
        'conv1': {
            'in_channels': 1,
            'out_channels': 10,
            'kernel_size': 5,
            'stride': 1,
            'padding': 1
            },
        'conv2': {
            'in_channels': 10,
            'out_channels': 20,
            'kernel_size': 5,
            'stride': 1,
            'padding': 1
            }
    }

poolLayers = {
    'pool1': {
        'kernel_size': 2
        },
    'pool2': {
        'kernel_size': 2
        }
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
        CNN_model = CNN.CNN_Class()
        
        # construct layers
        CNN_model.setLayers(imgSize=28, 
                            convLayer=convLayers, 
                            poolLayer=poolLayers, 
                            feature_out=10, 
                            n_hUnit=param_1, 
                            n_hLayer=1,
                            # printToggle=True
                            )
        
        # set parameters
        CNN_model.setParams(dropout_rate=0,
                            learning_rate=0.01,
                            act_lib='torch',
                            conv_activation_fun='relu',  # activation function at convolution layers (fully-connected layers)
                            pool_fun='max_pool2d',
                            activation_fun='relu',  # activation function at hidden layers (fully-connected layers)
                            optim_fun=param_0,
                            )

        dataloader_dict = {
            'DataLoader_train': DataLoader_train,
            'DataLoader_test': DataLoader_dev,
            }

        # result during training per each epoch
        _, model_losses_train, model_acc_ep_train, model_acc_ep_test, _, model_w_histx, model_w_histy = CNN_model.trainModel(
            **dataloader_dict,
            epochs=epochs,
            loss_function='cross-entropy',
            comp_acc_test=True,
            comp_w_hist=True,
            comp_w_change=False
        )

        res_losses_train[i, :] = model_losses_train.detach().numpy()

        res_acc_ep_train[i, :] = model_acc_ep_train.detach().numpy()

        # res_y_pred_train[:, :, i] = model_y_pred_train.detach().numpy()

        # accuracy comparing to test_set per each epoch
        res_acc_ep_test[i, :] = model_acc_ep_test.detach().numpy()

        # histogram weight
        res_w_histy[:, :, i] = model_w_histy

        # model parameters
        # nParams[e, i] = model_nParams

        # process time
        time_proc[e, i] = time.process_time() - time_start
        
        
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
        bestModel['net'] = copy.deepcopy(CNN_model.state_dict())
        bestModel['params'] = (e, i)

#%% check total number of parameters in the model
summary(CNN_model, (1, 28, 28))

#%% explore the model's weight each layer

net_input_layer = vars(CNN_model.layers['input'])
net_input_layer_weight = CNN_model.layers['input'].weight

# plot histogram of weight
plt.figure()
plt.hist(net_input_layer_weight.detach().flatten(), 40)
plt.xlabel('Weight value')
plt.ylabel('Count')
plt.title('Distribution of initialized weight at input layer')


# weight at all layers
W = np.array([])

# get set of weights from each layer
for layer in CNN_model.layers:
    W = np.concatenate((W, CNN_model.layers[f'{layer}'].weight.detach().flatten().numpy() ))

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
# t, p = stats.ttest_ind(y_plot_1, y_plot_2)
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
# t, p = stats.ttest_ind(y_plot_1, y_plot_2)
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.plot(np.arange(1, epochs+1), y_plot_2, 'k', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Accuracy by Epoch(test set)')
# plt.legend([f'{param}' for param in model_params_1] + ['Mean'])
# plt.legend(['Kaiming', 'Xavier'])


#%% Prediction from trained model

# run the model through for the test data
X = data_test_ind_ts
y = y_test_ind_ts
predictions, _ = CNN_model.predict(X, y)
predictions_raw = predictions.clone()
predictions = torch.softmax(predictions, axis=1)
predictions = torch.argmax(predictions, axis=1)

pname = 'Number'

# and plot those against the real data
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(predictions, 'o', label='Predicted values')
plt.plot(y+.2, 's', label='True values')
plt.xlabel('Stimulus number')
plt.ylabel('Category')
plt.yticks(np.unique(labels))
# plt.ylim()
plt.legend()
plt.title(f'Prediction - {pname}')

# accuracy by category
accuracy = (predictions == y).float()

# compute overall accuracy
totalAcc = torch.mean(100*accuracy).item()

# and average by group
accuracyByGroup = np.zeros(len(np.unique(labels)))
for i in np.unique(labels):
    accuracyByGroup[int(i)] = 100*torch.mean(accuracy[y == i])

plt.subplot(1, 2, 2)
plt.bar(np.unique(labels), accuracyByGroup)
# plt.ylim([80, 100])
plt.xticks(np.unique(labels))
plt.xlabel('Group')
plt.ylabel('Accuracy (%)')
plt.title(f'Final avg accuracy = {totalAcc:.2f}% - {pname}')


# find the errors
errors = np.where( predictions != y )[0]

# Evidence for all numbers from one sample
sample2show = np.random.randint(0, len(errors))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Example of error')

ax[0].bar(np.unique(labels), torch.softmax(predictions_raw, axis=1)[errors[sample2show]])
ax[0].set_xticks(np.unique(labels))
ax[0].set_xlabel('Number')
ax[0].set_ylabel('Evidence for that number')
ax[0].set_title('True number: %s, model guessed %s' 
                %( y[errors[sample2show]].item(), predictions[errors[sample2show]].item() ))

ax[1].imshow( np.reshape(X[errors[sample2show], :], (28, 28)), cmap='gray')

plt.show()
