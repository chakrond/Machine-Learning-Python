# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 23:11:49 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
import math
import time
import random
import os
import re

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=2)

#%% Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# sc_y = StandardScaler() # if y is already between 0 and 1, no need to scale
scaled_X_train = sc_X.fit_transform(X_train)
scaled_X_test = sc_X.transform(X_test) # **only transfrom for X_test prevent information leakage
# scaled_y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

#%% SVC Model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(scaled_X_train, y_train)

#%% k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print('Accuracy: {:.2f} %'.format(acc.mean()*100))
print('Standard Deviation: {:.2f} %'.format(acc.std()*100))

#%% Grid Search
from sklearn.model_selection import GridSearchCV
params =    [
                {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
                {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
            ]
          
grid_search_model = GridSearchCV(estimator=classifier, 
                           param_grid=params,
                           scoring='accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search_model.fit(X_train, y_train)
best_acc = grid_search_model.best_score_
best_params = grid_search_model.best_params_
print('Best Accuracy: {:.2f} %'.format(best_acc*100))
print('Best Parameters: ', best_params)