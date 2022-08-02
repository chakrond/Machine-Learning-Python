# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 23:11:43 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import random
import os

# TensorFlow
# import tensorflow as tf

# PyTorch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)

#%% Importing the dataset

# ------------------------------------------------------------------------ 
# Data set
# ------------------------------------------------------------------------ 
dataset_movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, 
                             engine='python', encoding='latin-1')
dataset_movies.columns = ['Movie ID', 'Title', 'Genre']

dataset_users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, 
                             engine='python', encoding='latin-1')
dataset_users.columns = ['User ID', 'Sex', 'Age', 'Jobs', 'Zip Code']

dataset_ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, 
                             engine='python', encoding='latin-1')
dataset_ratings.columns = ['User ID', 'Movie ID', 'Rating', 'TimeStamp']

#%% Dataset

# ------------------------------------------------------------------------ 
# Trainging set
# ------------------------------------------------------------------------ 
set_training = pd.read_csv('ml-100k/u1.base', delimiter='\t', header=None)
set_training.columns = ['User ID', 'Movie ID', 'Rating', 'Timestamp']
ary_set_training = np.array(set_training, dtype='int')

# ------------------------------------------------------------------------ 
# Test set
# ------------------------------------------------------------------------ 
set_test = pd.read_csv('ml-100k/u1.test', delimiter='\t', header=None)
set_test.columns = ['User ID', 'Movie ID', 'Rating', 'Timestamp']
ary_set_test = np.array(set_test, dtype='int')

#%% Data preprocessing

# Max userID and movieID
max_userID = max([max(ary_set_training[:, 0]), max(ary_set_test[:, 0])])
max_movieID = max([max(ary_set_training[:, 1]), max(ary_set_test[:, 1])])

def reconstruct(array_data):

    uID_mID_R_list = []
    
    for uID in range(1, max_userID + 1):
        
        uID_mID_R = np.zeros(max_movieID)
        user_mID = array_data[:, 1] [array_data[:, 0] == uID]
        user_R = array_data[:, 2] [array_data[:, 0] == uID]
        uID_mID_R[user_mID-1] = user_R
        uID_mID_R_list.append(list(uID_mID_R))
        
    return uID_mID_R_list

# List of list data
uID_mID_R_traning = reconstruct(ary_set_training)
uID_mID_R_test = reconstruct(ary_set_test)     

#%% List to Torch tensors
tensors_training = torch.FloatTensor(uID_mID_R_traning)
tensors_test = torch.FloatTensor(uID_mID_R_test)

# Convert rating to binary 0 - not liked , 1 - liked
def convert_bin(tensors):
    tensors[tensors == 0] = -1
    tensors[tensors == 1] = 0
    tensors[tensors == 2] = 0
    tensors[tensors >= 3] = 1
    return tensors
    
# Tensors dataset
tensors_training = convert_bin(tensors_training)
tensors_test = convert_bin(tensors_test)

#%% Restricted Boltzmann Machines(RBM) Class

# Reference: An Introduction to Restricted Boltzmann Machines - p23

class RBM():
    
    def __init__(self, nV, nH): # number of nV(V = visble nodes), number of nH(H = Hidden nodes)
        self.W = torch.randn(nH, nV) # initialize random weight in size (nH, nV)
        self.c = torch.randn(1, nH) # create bias of hidden nodes, 1 batch of 2d tensors 
        self.b = torch.randn(1, nV) 
    
    # sigmoid activation function - p24
    def sample_H(self, x): # Sample for Hidden nodes, x = visible neuron
        Wx = torch.mm(x, self.W.t()) # self.W.t = transpose of self.W, matrix multiplication of the matrices
        activation = Wx + self.c.expand_as(Wx)
        prob_H_given_V = torch.sigmoid(activation) # probability of hidden nodes given visble nodes
        
        return prob_H_given_V, torch.bernoulli(prob_H_given_V)
    
    def sample_V(self, y): # Sample for Visible nodes, y = visible neuron
        Wy = torch.mm(y, self.W)
        activation = Wy + self.b.expand_as(Wy)
        prob_V_given_H = torch.sigmoid(activation)
        
        return prob_V_given_H, torch.bernoulli(prob_V_given_H)
    
    def train(self, V0, Vk, PH0, PHk): # k step, 8, 9, 10 - p28
        # update values from step 8
        self.W += (torch.mm(V0.t(), PH0) - torch.mm(Vk.t(), PHk)).t() # step 8
        self.b += torch.sum((V0 - Vk), 0) # step 9, 0 is to keep the tensors format
        self.c += torch.sum((PH0 - PHk), 0) # step 10
        
#%% RBM Parameters

# Number of movies, nV (a visible nodes) 1 node for each movie, fixed number
nv = len(tensors_training[0])

# Number of hidden nodes, the desired number of features to be detected
nh = 100

# batch size (each 100 users)
batch_size = 100

#%RBM Model
rbm = RBM(nv, nh)

#%% Training RBM
n_epoch = 10

for epoch in range(0, n_epoch+1):
    loss_train = 0
    counter = 0.
    
    for uID in range(0, max_userID - batch_size, batch_size):
        vk = tensors_training[uID:(uID + batch_size)] # change through loop
        v0 = tensors_training[uID:(uID + batch_size)] # target not change
        ph0, _ = rbm.sample_H(v0)
        
        # Contrastive Divergence
        # k-step random walk Gibbs sampling, Markov Chain Monte Carlo Technique(MCMC)
        for k in range(10):
            _, hk = rbm.sample_H(vk) # k step 5
            _, vk = rbm.sample_V(hk) # k step 6
            vk[v0<0] = v0[v0<0]
            
        # get prob_H_given_V
        phk, _ = rbm.sample_H(vk)
        
        # update values, train
        rbm.train(v0, vk, ph0, phk)
        
        # update loss of train
        
        # Average Distance
        loss_train += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        
        # Root Mean Squared Error
        # loss_train += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE
        
        counter += 1
        nml_loss = np.around(loss_train/counter, decimals=4)
        
    print(f'epoch: {epoch}, train loss: {str(nml_loss)}')

#%% Testing RBM

loss_test = 0
counter = 0.
for uID in range(max_userID):
    v = tensors_training[uID:uID+1]
    vt = tensors_test[uID:uID+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_H(v)
        _,v = rbm.sample_V(h)
        
        # Average Distance
        loss_test += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        
        # Root Mean Squared Error
        # loss_test += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE
        
        counter += 1.
print('test loss: '+str(loss_test/counter))