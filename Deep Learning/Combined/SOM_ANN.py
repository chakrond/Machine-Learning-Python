# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 22:21:55 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import math
import time
import random
import os

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)

#%% Importing the dataset

# ------------------------------------------------------------------------ 
# Data set
# ------------------------------------------------------------------------ 
dataset = pd.read_csv('Credit_Card_Applications.csv')
set_ID = dataset.iloc[:, [0]].values
set_features = dataset.iloc[:, 1:-1].values
set_true_class = dataset.iloc[:, -1]

#%% Feature Scaling

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ------------------------------------------------------------------------ 
# Standardization (STD)
# ------------------------------------------------------------------------ 
# sc_X = StandardScaler()
# sc_y = StandardScaler() # if y is already between 0 and 1, no need to scale
# scaled_X_train = sc_X.fit_transform(X_train)
# scaled_X_test = sc_X.transform(X_test) # **only transfrom for X_test prevent information leakage
# scaled_y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

# ------------------------------------------------------------------------ 
# Scaling - Normalization
# ------------------------------------------------------------------------ 
sc_Norm = MinMaxScaler(feature_range=(0, 1))
scaled_set_features = sc_Norm.fit_transform(set_features)

#%% SOM Model
from minisom import MiniSom

SOM_model = MiniSom(x=10, y=10, input_len=scaled_set_features.shape[1], sigma=1.0, learning_rate=0.5) # size of grid X by y
SOM_model.random_weights_init(scaled_set_features)
SOM_model.train_random(data=scaled_set_features, num_iteration=500)

#%% Plot
from pylab import bone, pcolor, colorbar, plot, show


plt.figure(0)
bone()
distance = SOM_model.distance_map()
distance_T = distance.T
pcolor(distance_T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

winning_node_coord_sample = SOM_model.winner(scaled_set_features[0])
offset_param = 0.5

for i, n in enumerate(scaled_set_features):
    winning_node_coord = SOM_model.winner(n)
    
    # plot marker for ture class
    plot(winning_node_coord[0] + offset_param, winning_node_coord[1] + offset_param,
         markers[set_true_class[i]], markeredgecolor=colors[set_true_class[i]],
         markerfacecolor = 'None' , markersize=10, markeredgewidth=2)

    
#%% Find winning nodes
map_winning_nodes = SOM_model.win_map(scaled_set_features)

# Observe Single node (Max node)
observe_idx_winning_node = np.unravel_index(indices=distance_T.argmax(), shape=distance_T.shape, order='F')
observe_winning_node = map_winning_nodes[observe_idx_winning_node]
observe_winning_node_value = sc_Norm.inverse_transform(observe_winning_node)

# Observe Multiple nodes
mat = np.sort(np.ravel(distance_T), axis=0)
max_val = mat[-5:] # 3 max values
max_coord = []
# max_winning_node_value
for i in range(len(max_val)):
    idx = np.where(distance_T == max_val[i])
    max_coord.append((idx[0][0], idx[1][0]))
    
    if len(map_winning_nodes[max_coord[i]]) == 0:
        continue

    max_winning_node_value = np.array(map_winning_nodes[max_coord[i]])
    
    if len(map_winning_nodes[max_coord[i]]) > 1:
        np.concatenate((max_winning_node_value, map_winning_nodes[max_coord[i]]), axis=0)
        
# get actual value
actual_max_winning_node_value = sc_Norm.inverse_transform(max_winning_node_value)

# looking up customer ID
idx_match = []
for i in range(0, actual_max_winning_node_value.shape[0]):

    sum_log = []
    
    for r in range(0, set_features.shape[0]):

        sum_log.append(sum(actual_max_winning_node_value[i, :] == set_features[r, :]))

    sum_log = np.array(sum_log)
    idx_match.append(sum_log.argmax())

max_winning_node_ID = set_ID[np.array(idx_match)]

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
  if dataset.iloc[i, 0] in max_winning_node_ID:
    is_fraud[i] = 1

#%% ANN Model 

#%% Splitting the dataset
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(set_features, set_true_class, test_size=0.2, random_state=1)
X_train = dataset.iloc[:, 1:].values
y_train = is_fraud

#%% Feature Scaling

sc = StandardScaler()
# sc = MinMaxScaler(feature_range=(0, 1))
X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

#%% Building ANN (Create Sequential model)
ann = tf.keras.models.Sequential() # Model

# Add input layer, 1st layer
ann.add(tf.keras.layers.Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=X_train.shape[1]))

# Add input layer, 2nd layer
ann.add(tf.keras.layers.Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Add output layer
ann.add(tf.keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training ANN
with tf.device('/cpu:0'):
# with tf.device('/gpu:0'):
    ann.fit(X_train, y_train, batch_size=32, epochs=100) # batch learning

# Make prediction
# y_pred = ann.predict(X_test) # result give the propability
# y_pred_bool = (y_pred>0.5)

# model = [f'c{i}' for i in range(0, n_model)]
# model = [eval(clf) for clf in classifiers]
models = [ann]

#%% Making confusion matrix   
# from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

# titles = ['ANN Model']

# n_model = len(titles)

# nrows = 2
# ncols = 1
# fig, axes = plt.subplots(nrows, ncols, figsize=(15,10))

# for i in range(0, nrows*ncols):
#     axes.flatten()[i].axis("off")
 
# for i, model, title in zip(np.arange(0, n_model), models, titles):
    
#     fig.add_subplot(axes.flatten()[i])
#     y_pred = model.predict(X_test)
#     y_pred_bool = (y_pred>0.5)
#     ConfusionMatrixDisplay.from_predictions(y_test, y_pred_bool, ax=axes.flatten()[i], colorbar=False)
#     acc_score = round(accuracy_score(y_test, y_pred_bool), 2)
#     axes.flatten()[i].title.set_text(f'{title}, Accuracy Score = {acc_score}')
#     plt.axis("on")
#     plt.tight_layout()
    
#%% Make prediction on Customer ID
customer_input = dataset.iloc[:, 1:].values
customer_pred = ann.predict(customer_input)
customer_prop_ID = np.concatenate((max_winning_node_ID, customer_pred[np.array(idx_match)]*100), axis=1)
