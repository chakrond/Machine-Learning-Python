# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:34:05 2022

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

# Settings
np.set_printoptions(precision=2)

#%% Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#%% Encoding categorical data
# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoder [Gender Column]
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Create Dummy Variables [Country]
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#%% Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Building ANN (Create Sequential model)
ann = tf.keras.models.Sequential() # Model

# Add input layer, 1st layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add input layer, 2nd layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Add output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training ANN
ann.fit(X_train, y_train, batch_size=32, epochs=100) # batch learning

# Make prediction
# y_pred = ann.predict(X_test) # result give the propability
# y_pred_bool = (y_pred>0.5)

# model = [f'c{i}' for i in range(0, n_model)]
# model = [eval(clf) for clf in classifiers]
models = [ann]
#%% Making confusion matrix   
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

titles = ['ANN Model']

n_model = len(titles)

nrows = 2
ncols = 1
fig, axes = plt.subplots(nrows, ncols, figsize=(15,10))

for i in range(0, nrows*ncols):
    axes.flatten()[i].axis("off")
 
for i, model, title in zip(np.arange(0, n_model), models, titles):
    
    fig.add_subplot(axes.flatten()[i])
    y_pred = model.predict(X_test)
    y_pred_bool = (y_pred>0.5)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_bool, ax=axes.flatten()[i], colorbar=False)
    acc_score = round(accuracy_score(y_test, y_pred_bool), 2)
    axes.flatten()[i].title.set_text(f'{title}, Accuracy Score = {acc_score}')
    plt.axis("on")
    plt.tight_layout()
