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

#%% XGBoost model
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#%% Confusion Matrix
from sklearn.metrics import  ConfusionMatrixDisplay, accuracy_score

y_pred = classifier.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False)
acc_score = round(accuracy_score(y_test, y_pred), 2)
plt.title(f'Confusion Matrix, ACC = {acc_score}')
plt.axis("on")
plt.xlabel('')
# plt.ylabel('')
plt.tight_layout()

#%% k-Fold Cross Validation
from sklearn.model_selection import cross_val_score

acc = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print('Accuracy: {:.2f} %'.format(acc.mean()*100))
print('Standard Deviation: {:.2f} %'.format(acc.std()*100))
