# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:52:28 2022

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
dataset = pd.read_csv('Wine.csv')
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

#%% Apply PCA 
from sklearn.decomposition import PCA
n_components = 2
pca = PCA(n_components=n_components)

pca_X_train = pca.fit_transform(scaled_X_train)
pca_X_test = pca.transform(scaled_X_test)

#%% Apply kernel PCA 
from sklearn.decomposition import KernelPCA
n_components = 2
kernel_pca = KernelPCA(n_components=n_components, kernel='rbf')

kernel_pca_X_train = kernel_pca.fit_transform(scaled_X_train)
kernel_pca_X_test = kernel_pca.transform(scaled_X_test)

#%% Apply LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

n_components = 2
LDA = LinearDiscriminantAnalysis(n_components=n_components)
LDA_X_train = LDA.fit_transform(scaled_X_train, y_train)
LDA_X_test = LDA.transform(scaled_X_test)

#%% Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Model without Dimensionality Reduction

# Title
title = ['Logistic', 'KNeighbors', 
         'SVC with Linear kernel', 'SVC with RBF kernel', 
         'Gaussian Naive Bayes', 'Decision Tree',
         'Random Forest']

X_train_set, y_train_set = scaled_X_train, y_train

clf_000 = LogisticRegression(random_state=0).fit(X_train_set, y_train_set)
clf_001 = KNeighborsClassifier(n_neighbors=5, metric='minkowski').fit(X_train_set, y_train_set)
clf_002 = SVC(kernel = 'linear', random_state=0).fit(X_train_set, y_train_set)
clf_003 = SVC(kernel = 'rbf', random_state=0).fit(X_train_set, y_train_set)
clf_004 = GaussianNB().fit(X_train_set, y_train_set)
clf_005 = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train_set, y_train_set)
clf_006 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0).fit(X_train_set, y_train_set)


# Model with PCA

# Title
title_PCA = ['Logistic - PCA', 'KNeighbors - PCA', 
             'SVC with Linear kernel - PCA', 'SVC with RBF kernel - PCA', 
             'Gaussian Naive Bayes - PCA', 'Decision Tree - PCA',
             'Random Forest - PCA']

X_train_set, y_train_set = pca_X_train, y_train

clf_100 = LogisticRegression(random_state=0).fit(X_train_set, y_train_set)
clf_101 = KNeighborsClassifier(n_neighbors=5, metric='minkowski').fit(X_train_set, y_train_set)
clf_102 = SVC(kernel = 'linear', random_state=0).fit(X_train_set, y_train_set)
clf_103 = SVC(kernel = 'rbf', random_state=0).fit(X_train_set, y_train_set)
clf_104 = GaussianNB().fit(X_train_set, y_train_set)
clf_105 = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train_set, y_train_set)
clf_106 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0).fit(X_train_set, y_train_set)


# Model with kernel PCA

# Title
title_kPCA = ['Logistic - kPCA', 'KNeighbors - kPCA', 
              'SVC with Linear kernel - kPCA', 'SVC with RBF kernel - kPCA', 
              'Gaussian Naive Bayes - kPCA', 'Decision Tree - kPCA',
              'Random Forest - kPCA']

X_train_set, y_train_set = kernel_pca_X_train, y_train

clf_200 = LogisticRegression(random_state=0).fit(X_train_set, y_train_set)
clf_201 = KNeighborsClassifier(n_neighbors=5, metric='minkowski').fit(X_train_set, y_train_set)
clf_202 = SVC(kernel = 'linear', random_state=0).fit(X_train_set, y_train_set)
clf_203 = SVC(kernel = 'rbf', random_state=0).fit(X_train_set, y_train_set)
clf_204 = GaussianNB().fit(X_train_set, y_train_set)
clf_205 = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train_set, y_train_set)
clf_206 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0).fit(X_train_set, y_train_set)


# Model with LDA

# Title
title_LDA = ['Logistic - LDA', 'KNeighbors - LDA', 
              'SVC with Linear kernel - LDA', 'SVC with RBF kernel - LDA', 
              'Gaussian Naive Bayes - LDA', 'Decision Tree - LDA',
              'Random Forest - LDA']

X_train_set, y_train_set = LDA_X_train, y_train

clf_300 = LogisticRegression(random_state=0).fit(X_train_set, y_train_set)
clf_301 = KNeighborsClassifier(n_neighbors=5, metric='minkowski').fit(X_train_set, y_train_set)
clf_302 = SVC(kernel = 'linear', random_state=0).fit(X_train_set, y_train_set)
clf_303 = SVC(kernel = 'rbf', random_state=0).fit(X_train_set, y_train_set)
clf_304 = GaussianNB().fit(X_train_set, y_train_set)
clf_305 = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train_set, y_train_set)
clf_306 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0).fit(X_train_set, y_train_set)

#%% Generate Model List
n_model = len(title)
classifiers_name =  [f'clf_00{i}' for i in range(0, n_model)] + [f'clf_10{i}' for i in range(0, n_model)] + \
                    [f'clf_20{i}' for i in range(0, n_model)] + [f'clf_30{i}' for i in range(0, n_model)]

classifiers = [eval(clf) for clf in classifiers_name]
titles_set  = [title, title_PCA, title_kPCA, title_LDA]

#%% Confusion matrix   
from sklearn.metrics import  ConfusionMatrixDisplay, accuracy_score

X_set_name = ['scaled_X_', 'pca_X_', 'kernel_pca_X_', 'LDA_X_']
X_set_name_train = [name + 'train' for name in X_set_name]
X_set_name_test  = [name + 'test' for name in X_set_name]

nrows = 4
ncols = 7
fig, axes = plt.subplots(nrows, ncols, figsize=(6,6))
plt.rcParams['font.size'] = '6'
fig.suptitle('Confusion Matrix Comparison of Dimensionality Reduction',fontweight ="bold", fontsize=12)

for i in range(0, nrows*ncols):
    axes.flatten()[i].axis("off")
 
idx_subplot = 0
for (m, x_set_name_type) in enumerate(X_set_name_test): 

    X_confMat_model, y_confMat_model = eval(x_set_name_type), y_test
    
    # Filter the model number and name
    model_list      = list(filter(lambda elem: re.match(f'^clf_{m}', elem[1]), enumerate(classifiers_name)))
    name_model_list = [idx[1] for idx in model_list]
    idx_model_list  = [idx[0] for idx in model_list]
    

    for (i, k) in enumerate(idx_model_list):
        
        fig.add_subplot(axes.flatten()[idx_subplot])
        y_pred = classifiers[k].predict(X_confMat_model)
        ConfusionMatrixDisplay.from_predictions(y_confMat_model, y_pred, ax=axes.flatten()[idx_subplot], colorbar=False)
        acc_score = round(accuracy_score(y_confMat_model, y_pred), 2)
        axes.flatten()[idx_subplot].title.set_text(f'{titles_set[m][i]}, ACC = {acc_score}')
        plt.axis("on")
        # plt.xlabel('')
        # plt.ylabel('')
        plt.tight_layout()
        idx_subplot += 1

#%% Create a mesh to plot, Decision boundary (Training Set)

from matplotlib.colors import ListedColormap

X_set_name = ['pca_X_', 'kernel_pca_X_', 'LDA_X_']
X_set_name_train = [name + 'train' for name in X_set_name]
X_set_name_test  = [name + 'test' for name in X_set_name]

fig = plt.figure(2)
fig.suptitle('Decision Boundary Comparison [Training Set]', fontweight ='bold')

# Plot decision boundary

# Training set data
idx_subplot = 1
for (m, x_set_name_type) in zip([1, 2, 3], X_set_name_train): # Exclude normal model

    X_set, y_set = eval(x_set_name_type), y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.5),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.5))
    
    # Filter the model number and name
    model_list      = list(filter(lambda elem: re.match(f'^clf_{m}', elem[1]), enumerate(classifiers_name)))
    name_model_list = [idx[1] for idx in model_list]
    idx_model_list  = [idx[0] for idx in model_list]
    
    for (i, k) in enumerate(idx_model_list):

        plt.subplot(3, 7, idx_subplot)
        plt.rcParams['font.size'] = '6'
        idx_subplot += 1
        Z = classifiers[k].predict(np.array([X1.ravel(), X2.ravel()]).T)
        # Put the result into a color plot
        Z = Z.reshape(X1.shape)
        # plt.contourf(X1, X2, Z, cmap=plt.cm.Paired)
        plt.contourf(X1, X2, Z, alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
        
        # Title  
        plt.title(f'{titles_set[m][i]}')
        
        # plt.xlabel('xlabel', fontsize=10)
        # plt.ylabel('ylabel', fontsize=10)

        # plt.axis("off")
        
        # Plot the training points
        for (q, j) in enumerate(np.unique(y_set)):
            plt.scatter(X_set[:, 0].reshape((len(X_set), 1))[y_set == j], X_set[:, 1].reshape((len(X_set), 1))[y_set == j], s = 20, c = ListedColormap(('red', 'green', 'blue'))(q), label = j)
            # plt.xlabel('Age')
            # plt.ylabel('Estimated Salary')
            plt.legend()
        