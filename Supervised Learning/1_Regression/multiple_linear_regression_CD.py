# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 00:39:10 2022

@author: Chakron.D
"""
# Checking directory
import os
os.getcwd()
os.listdir()


# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(f'X = {X}')
print(f'y = {y}')

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(f'ColumnTransformer = {X}')

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Simple Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # train network

# predicting the test results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print('y_pred, y_test = ', np.concatenate( (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1) )

# Plot dataset
for i in range(3, len(X_train)):
    plt.figure(0)
    plt.scatter(X_train[:, i], y_train)


