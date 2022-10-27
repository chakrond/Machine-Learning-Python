# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 22:50:23 2022

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
np.set_printoptions(precision=2)

#%% Importing the dataset

# ------------------------------------------------------------------------ 
# Training set
# ------------------------------------------------------------------------ 
dataset = pd.read_csv('Google_Stock_Price_Train.csv')
set_traning = dataset.iloc[:, [1]].values

# ------------------------------------------------------------------------ 
# Test set
# ------------------------------------------------------------------------ 
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
stock_real = dataset_test.iloc[:, [1]].values

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
scaled_set_traning = sc_Norm.fit_transform(set_traning)

#%% Data Structure

time_step = 60

# ------------------------------------------------------------------------ 
# Training set
# ------------------------------------------------------------------------ 
X_train = []
y_train = []

for i in range(time_step, len(scaled_set_traning)):
    X_train.append(scaled_set_traning[(i-time_step):i, 0])
    y_train.append(scaled_set_traning[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape to 3 dimensions
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # reshape to 3 dimension (batch_size, time_step, indicator(number of set of information))

# ------------------------------------------------------------------------ 
# Total dataset
# ------------------------------------------------------------------------ 
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis=0, ignore_index=True)

# ------------------------------------------------------------------------ 
# Test set
# ------------------------------------------------------------------------ 
dataset_inputs = dataset_total[ (len(dataset_total) - len(dataset_test)) - time_step :].values.reshape(-1, 1)
scaled_inputs = sc_Norm.transform(dataset_inputs)

X_test = []

for i in range(time_step, len(scaled_inputs)):
    X_test.append(scaled_inputs[(i-time_step):i, 0])
    
X_test = np.array(X_test)

# Reshape to 3 dimensions
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) 

#%% RNN Model

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Sequential Model instance
regressor_model_RNN = Sequential()

# ------------------------------------------------------------------------ 
# No.1 LSTM Layer and Dropout
# ------------------------------------------------------------------------
regressor_model_RNN.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
        )
    )

regressor_model_RNN.add(Dropout(0.2)) # dropout rate is 20%

# ------------------------------------------------------------------------ 
# No.2 LSTM Layer and Dropout 
# ------------------------------------------------------------------------
regressor_model_RNN.add(
    LSTM(
        units=50,
        return_sequences=True
        )
    )

regressor_model_RNN.add(Dropout(0.2))

# ------------------------------------------------------------------------ 
# No.3 LSTM Layer and Dropout 
# ------------------------------------------------------------------------
regressor_model_RNN.add(
    LSTM(
        units=50,
        return_sequences=True
        )
    )

regressor_model_RNN.add(Dropout(0.2))

# ------------------------------------------------------------------------ 
# No.4 LSTM Layer and Dropout 
# ------------------------------------------------------------------------
regressor_model_RNN.add(
    LSTM(
        units=50
        )
    )

regressor_model_RNN.add(Dropout(0.2))

# ------------------------------------------------------------------------
# Output layer
# ------------------------------------------------------------------------
regressor_model_RNN.add(
    Dense(
        units=1, # one output
        )
    )

# ------------------------------------------------------------------------
# Compile
# ------------------------------------------------------------------------
regressor_model_RNN.compile(
    optimizer='adam',
    loss='mean_squared_error'
    )

# ------------------------------------------------------------------------
# Training the model
# ------------------------------------------------------------------------
regressor_model_RNN.fit( # batch learning
    X_train, 
    y_train, # True value
    epochs=100,
    batch_size=32, # to be trained/update at every 32 values
    # steps_per_epoch=X_train.shape[0], # batch_size
    # validation_steps=800 # batch_size
    ) 

# save weights of the model
regressor_model_RNN.save_weights('LSTM_first_try.h5')  # save weights after training or during training

#%% Make a prediction

stock_pred = regressor_model_RNN.predict(X_test)
stock_pred = sc_Norm.inverse_transform(stock_pred)

# Mean squared error(MSE)
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(stock_real, stock_pred))

#%% Plot
plt.rcParams['font.size'] = '10'
plt.plot(stock_real, c='r', label = 'Real Stock Price')
plt.plot(stock_pred, c='b', label = 'Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()