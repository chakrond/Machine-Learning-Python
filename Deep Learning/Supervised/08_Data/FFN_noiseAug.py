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
import Class.FFN as FFN

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

#%% Import dataset

# MNIST data
fname = 'z:\python traning\my codes\machine learning\deep learning\supervised\99_Datasets\MNIST\mnist_train_small.csv'
fdata = np.loadtxt(open(fname, 'rb'), delimiter=',')

# extract data and labels
labels = fdata[:,0]
data = fdata[:,1:]

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
# dc = data.describe()

# # list number of unique values per column
# for i in data.keys():
#   print(f'{i} has {len(np.unique(data[i]))} unique values')
  
# # pairwise plots
# cols2plot = ['fixed acidity', 'volatile acidity', 'citric acid', 'quality']
# sns.pairplot(data[cols2plot], kind='reg', hue='quality')
# plt.show()

# # boxplot
# fig, ax = plt.subplots(1, figsize=(17, 4))
# ax = sns.boxplot(data=data)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# ax.set_title('Data values Distribution')
# plt.show()


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

#%% data preprocessing

# # z-score all variables except for quality label

# # copy
# data_z = data.copy()

# # data features column name
# features = data_z.keys() #.drop('quality')

# # z-score
# for col in features:
#     meanval   = np.mean(data_z[col])
#     stdev     = np.std(data_z[col], ddof=1)
#     data_z[col] = (data_z[col] - meanval) / stdev

# # z-score by scipy
# # data_z[features] = data_z[features].apply(stats.zscore)

# dc_z = data_z.describe()

# # boxplot
# fig, ax = plt.subplots(1, figsize=(17,4))
# ax = sns.boxplot(data=data_z)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# plt.show()

# # inspect values
# plt.figure()
# plt.plot(data_z[value_name].values,'s')
# plt.xlabel('Data observation')
# plt.ylabel(f'{value_name} (normalized)')
# plt.show()

# # create a new column for binarized boolean quality [0 or 1]
# # data_z['boolQuality'] = 0
# # data_z['boolQuality'][data_z['quality']>5] = 1 # wine quality > 5 will assigned as 1
# # data_z[['quality', 'boolQuality']]

# # Inspect the correlation matrix
# fig = plt.figure(figsize=(8,8))

# corrcoef_mat = np.corrcoef(data_z.T)
# plt.imshow(corrcoef_mat, vmin=-0.3, vmax=0.3)
# plt.xticks(range(len(data_z.keys())), labels=data_z.keys(), rotation=90)
# plt.yticks(range(len(data_z.keys())), labels=data_z.keys())
# plt.colorbar()
# plt.title('Data correlation matrix')
# plt.show()

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

# # binarize data
# dataBin = data / np.max(data)
# dataBin[dataBin>0] = 1

# fig,ax = plt.subplots(1, 2, figsize=(10, 4))
# ax[0].hist(data.flatten(), 50)
# ax[0].set_xlabel('Pixel intensity values')
# ax[0].set_ylabel('Count')
# ax[0].set_title('Histogram of original data')
# ax[0].set_yscale('log')

# ax[1].hist(dataBin.flatten(), 50)
# ax[1].set_xlabel('Pixel intensity values')
# ax[1].set_ylabel('Count')
# ax[1].set_title('Histogram of binalized data')
# ax[1].set_yscale('log')

# plt.show()


# # example binarized 5s 

# # find indices of all the 5's in the dataset
# idx = np.where(labels==5)[0]

# # draw the first 12
# fig,axs = plt.subplots(2, 6, figsize=(15, 6))

# for i,ax in enumerate(axs.flatten()):
#   img = np.reshape(dataBin[idx[i], :], (28, 28))
#   ax.imshow(img, cmap='gray')
#   ax.axis('off')

# plt.suptitle("Example 5's", fontsize=20)
# plt.tight_layout(rect=[0, 0, 1, .95])

#%% convert to tensor

# divide independent set
data_ss, data_test_ind, y_ss, y_test_ind = train_test_split(dataNorm, labels, test_size=0.1)


# independent test set
# convert to tensor
data_test_ind_ts = torch.tensor(data_test_ind).float()
y_test_ind_ts = torch.tensor(y_test_ind).long()

# convert to dataset and dataLoader
Dataset_test = TensorDataset(data_test_ind_ts, y_test_ind_ts)
DataLoader_test = DataLoader(Dataset_test, batch_size=Dataset_test.tensors[0].shape[0])


# train set and dev set
data_ts = torch.tensor(data_ss).float()
y_ts = torch.tensor(y_ss).long()
# y_ts = y_ts.reshape((-1, 1))

#%% augment noise data

# gaussain noise
dataNorm_std = np.std(dataNorm)
dataNorm_mean = np.mean(dataNorm)
gs_n = np.random.normal(dataNorm_mean, dataNorm_std, dataNorm.shape) # gaussain distribution

plt.subplot(1, 3, 1)
img = np.reshape( dataNorm[-1, :], (28, 28))
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(1, 3, 2)
img_gs_n = np.reshape( gs_n[-1, :], (28, 28))
plt.imshow(img_gs_n, cmap='gray')
plt.title('Gaussian noise')

plt.subplot(1, 3, 3)
img_aug = np.reshape( dataNorm[-1, :] + gs_n[-1, :], (28, 28))
plt.imshow(img_aug, cmap='gray')
plt.title('Noise augmented')


# noise augmentation
noise_na = noise(data_ss, 2)
dataNorm_na = np.concatenate((data_ss, noise_na))
labels_na = np.concatenate((y_ss, y_ss))
# example augmented data
plt.figure()
img = np.reshape( dataNorm_na[-1, :], (28, 28))
plt.imshow(img, cmap='gray')
# noise augmented data
data_ts_na = torch.tensor(dataNorm_na).float()
y_ts_na = torch.tensor(labels_na).long()

# %% train/test dataset

# params
test_size = 0.2
b_size = 32

# train and dev set
DataLoader_train, DataLoader_dev, size = dSplit(data_ts, y_ts, model_test_size=test_size, p_batch_size=b_size)


# noise augmented set
DataLoader_train_na, DataLoader_dev_na, _ = dSplit(data_ts_na, y_ts_na, model_test_size=test_size, p_batch_size=b_size)


# data size summary
d = {'Name': ['Total data', 'Training data', 'Devset data', 'Test data'],
     'Size': [data.shape, size[0], size[1], data_test_ind_ts.shape]}

size_table = pd.DataFrame(data=d)


# check data
# note: observe dataloader by iterate through them
# for X, y in DataLoader_train:
#   print(X.shape, y.shape)
# X, y

# X, y = next(iter(DataLoader_train))
# X, y


# %% Create/Train model

# experiment parameters
optimTypes = ['Adam']
dataLoaderSet = [[DataLoader_train, DataLoader_dev], [DataLoader_train_na, DataLoader_dev_na]]

model_params_0 = optimTypes
model_params_1 = dataLoaderSet
# model_params_name   = ['Number of Hidden Layer', '']

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
        FFN_model = FFN.FFN_Class()
        FFN_model.setParams(feature_in=data_ts.shape[1],
                                 feature_out=10,
                                 n_hUnit=32,
                                 n_hLayer=2,
                                 dropout_rate=dr,
                                 learning_rate=lr,
                                 #w_decay='L2',
                                 #p_lambda=param_1,
                                 #p_momentum=0.8,
                                 batch_norm=None,
                                 act_lib='torch',
                                 activation_fun='relu',
                                 optim_fun=param_0,
                                 # lr_decay=param_1,
                                 # lr_step_size=p_batch_size*len(DataLoader_train),
                                 # lr_gamma=0.5,
                                 )

        # dict
        dataloader_dict = {
            'DataLoader_train': param_1[0],
            'DataLoader_test': param_1[1],
            }

        # result during training per each epoch
        _, model_losses_train, model_acc_ep_train, model_acc_ep_test, _, model_w_histx, model_w_histy = FFN_model.trainModel(
            **dataloader_dict,
            epochs=epochs,
            loss_function='NLL',
            comp_acc_test=True,
            comp_w_hist=True,
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
        bestModel['net'] = copy.deepcopy(FFN_model.state_dict())
        bestModel['params'] = (e, i)

#%% explore the model's parameters

params_table = pd.DataFrame()
params_table['Name'] = []
params_table['Weight'] = []
params_table['accumWeight'] = []

n_weights = 0
i = 0
# count weight except bias term
for param_name, weight in FFN_model.named_parameters():
    if 'bias' not in param_name:
        n_weights = n_weights + weight.numel()
        params_table.loc[i, 'Name'] = param_name
        params_table.loc[i, 'Weight'] = weight.numel()
        params_table.loc[i, 'accumWeight'] = n_weights
        i += 1

#%% explore the model's layer

net_input_layer = vars(FFN_model.layers['input'])
net_input_layer_weight = FFN_model.layers['input'].weight

# plot histogram of weight
plt.figure()
plt.hist(net_input_layer_weight.detach().flatten(), 40)
plt.xlabel('Weight value')
plt.ylabel('Count')
plt.title('Distribution of initialized weight at input layer')


# weight at all layers
W = np.array([])

# get set of weights from each layer
for layer in FFN_model.layers:
    W = np.concatenate((W, FFN_model.layers[f'{layer}'].weight.detach().flatten().numpy() ))

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
plt.title('Histograms of weights')
plt.xlabel('Weight value')
plt.ylabel('Density')

#%% Model's weight histogram of each epoch

# show the histogram of the weights

fig,ax = plt.subplots(1, 3, figsize=(15, 5))

model_w_histy = list_w_histy_params_0[0][:, :, 1]

# # plot histogram of weight (bar)
# ax[0].hist(W, bins=100)
# ax[0].set_xlabel('Weight value')
# ax[0].set_ylabel('Count')
# ax[0].set_title('Distribution of weights at all layers')


# plot histogram of weight (line)
for i in range(model_w_histy.shape[0]-1):
    ax[1].plot(model_w_histx, model_w_histy[i, :], color=[1-i/100, 0.3, i/100], alpha=0.5)
ax[1].plot(model_w_histx, model_w_histy[-1, :], color=[1-model_w_histy.shape[0]/100, 0.3, model_w_histy.shape[0]/100], label='Final Weights')
ax[1].set_title('Histograms of weights')
ax[1].set_xlabel('Weight value')
ax[1].set_ylabel('Density')
ax[1].legend()

# plot image of weight with epoch
ax[2].imshow(model_w_histy, vmin=0, vmax=0.8,
             extent=[model_w_histx[0], model_w_histx[-1], 0, 99], aspect='auto', origin='upper', cmap='turbo')
ax[2].set_xlabel('Weight value')
ax[2].set_ylabel('Training epoch')
ax[2].set_title('Image of weight histograms')

plt.show()


#%% Predictionfrom trained model

# run the model through for the test data
# X, y = next(iter(DataLoader_test))
X, y = data_test_ind_ts, y_test_ind_ts
predictions, _ = FFN_model.predict(X, y)
predicitons = torch.log_softmax(predictions, axis=1)


# Evidence for all numbers from one sample
sample2show = np.random.randint(0, len(predicitons))

# plt.bar(range(10), predictions[sample2show])
plt.bar(range(10), torch.exp(predictions[sample2show]))
plt.xticks(range(10))
plt.xlabel('Number')
plt.ylabel('Evidence for that number')
plt.title(f'True number was {y[sample2show]}')
plt.show()


# find the errors
errors = np.where( torch.max(predictions, axis=1)[1] != y )[0]

# Evidence for all numbers from one sample
sample2show = np.random.randint(0, len(errors))

fig,ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].bar(range(10), np.exp(predictions[errors[sample2show]]))
ax[0].set_xticks(range(10))
ax[0].set_xlabel('Number')
ax[0].set_ylabel('Evidence for that number')
ax[0].set_title('True number: %s, model guessed %s' 
                %( y[errors[sample2show]].item(), torch.argmax(predictions[errors[sample2show]]).item() ))

ax[1].imshow( np.reshape(X[errors[sample2show], :], (28, 28)), cmap='gray')

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
y_plot_1 = np.array(list_acc_ep_train_params_0[0]).T
y_plot_2 = np.mean(np.array(list_acc_ep_train_params_0[0]), axis=0)
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.plot(np.arange(1, epochs+1), y_plot_2, 'k', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch(Train set)')
plt.legend([f'{param}' for param in model_params_1] + ['Mean'])

plt.subplot(1, 2, 2)
y_plot_1 = np.array(list_acc_ep_test_params_0[0]).T
y_plot_2 = np.mean(np.array(list_acc_ep_test_params_0[0]), axis=0)
plt.plot(np.arange(1, epochs+1), y_plot_1, linewidth=1)
plt.plot(np.arange(1, epochs+1), y_plot_2, 'k', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy by Epoch(test set)')
plt.legend([f'{param}' for param in model_params_1] + ['Mean'])

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
# get the categorical predictions

for (li, pname) in zip(list_y_pred_model_params_0, model_params_0):

    yHat = torch.tensor(np.mean(li, axis=0))
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
    plt.title(f'Prediction - {pname}')
    
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
    plt.title(f'Final accuracy = {totalAcc:.2f}% - {pname}')
    
    
    sm = nn.Softmax(dim=1)
    fig = plt.figure()
    fig.suptitle(f'Overall performance of model prediction - {pname}', fontweight='bold')
    
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

#%% acc vs learning rate by optimizer

# average over last 100 epochs
a_p_0 = np.zeros((len(model_params_0), len(model_params_1)))
a_p_1 = np.zeros((len(model_params_0), len(model_params_1)))

for i, lis in enumerate(list_acc_ep_train_params_0):
    a_p_0[i, :] = np.mean(lis[:, -100:], axis=1)
for i, lis in enumerate(list_acc_ep_test_params_0):
    a_p_1[i, :] = np.mean(lis[:, -100:], axis=1)

pName = 'nUnit'

plt.figure()

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# colors = ['orange', 'purple', 'green']
s_color = colors[0:len(model_params_0)]
plt.gca().set_prop_cycle(color=s_color)

plt.plot(model_params_1, a_p_0.T, 'o-')
plt.plot(model_params_1, a_p_1.T, 's--')
plt.xlabel('Experiment parameters')
plt.ylabel('Accuracy')
plt.legend([f'{param} - Train' for param in model_params_0] + [f'{param} - Test' for param in model_params_0])
plt.title('')
plt.xscale('log')


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
ax[0].set_xscale('log')

ax[1].plot(model_params_1, -np.diff(a_p.T, axis=1), 'o-')
# ax[1].plot([0, .9], [0, 0], 'k--')
ax[1].set_xlabel('Experiment Parameter last 100 epochs')
ax[1].set_ylabel('Train-test difference (acc%)')
ax[1].set_xscale('log')

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
ax[0].set_xscale('log')

ax[1].plot(model_params_1, -np.diff(a_p.T, axis=1), 'o-')
# ax[1].plot([0, .9], [0, 0], 'k--')
ax[1].set_xlabel('Experiment Parameter')
ax[1].set_ylabel('Train-test difference (acc%)')
ax[1].set_xscale('log')

#%% Save model

torch.save(FFN_model.state_dict(), 'trainedModel.pt')

#%% Load model

FFN_model = FFN.FFN_Class()
