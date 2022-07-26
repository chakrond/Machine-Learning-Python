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

# Title
title = ['Logistic', 'KNeighbors', 
         'SVC with Linear kernel', 'SVC with RBF kernel', 
         'Gaussian Naive Bayes', 'Decision Tree',
         'Random Forest']

X_train_set, y_train_set = LDA_X_train, y_train

clf_0 = LogisticRegression(random_state=0).fit(X_train_set, y_train_set)
clf_1 = KNeighborsClassifier(n_neighbors=5, metric='minkowski').fit(X_train_set, y_train_set)
clf_2 = SVC(kernel = 'linear', random_state=0).fit(X_train_set, y_train_set)
clf_3 = SVC(kernel = 'rbf', random_state=0).fit(X_train_set, y_train_set)
clf_4 = GaussianNB().fit(X_train_set, y_train_set)
clf_5 = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train_set, y_train_set)
clf_6 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0).fit(X_train_set, y_train_set)

n_model = n_model = len(title)
classifiers = [f'clf_{i}' for i in range(0, n_model)]
classifiers = [eval(clf) for clf in classifiers]

#%% Confusion matrix   
from sklearn.metrics import  ConfusionMatrixDisplay, accuracy_score

nrows = 3
ncols = 3

fig, axes = plt.subplots(nrows, ncols, figsize=(15,10))
fig.suptitle('Confusion Matrix Comparison',fontweight ="bold")


for i in range(0, nrows*ncols):
    axes.flatten()[i].axis("off")
 
for (i, clf) in enumerate(classifiers):
    
    fig.add_subplot(axes.flatten()[i])
    ConfusionMatrixDisplay.from_estimator(clf, LDA_X_test, y_test, ax=axes.flatten()[i], colorbar=False)
    y_pred = clf.predict(LDA_X_test)
    acc_score = round(accuracy_score(y_test, y_pred), 2)
    axes.flatten()[i].title.set_text(f'{title[i]}, Accuracy Score = {acc_score}')
    plt.axis("on")
    plt.tight_layout()
    
#%% Create a mesh to plot, Decision boundary (Training Set)
X_set, y_set = LDA_X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.5))


from matplotlib.colors import ListedColormap
# color_map = {-1: (1, 1, 1), 0: (0, 0, 0.9), 1: (1, 0, 0), 2: (0.8, 0.6, 0)}

fig = plt.figure(2)
fig.suptitle('Decision Boundary Comparison [Training Set]',fontweight ='bold')

# for i in range(0, len(classifiers)):
for (i, clf) in enumerate(classifiers):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.figure(1)
    plt.subplot(3, 3, i + 1)
    Z = clf.predict(np.array([X1.ravel(), X2.ravel()]).T)
    # Put the result into a color plot
    Z = Z.reshape(X1.shape)
    # plt.contourf(X1, X2, Z, cmap=plt.cm.Paired)
    plt.contourf(X1, X2, Z, alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    
    # plt.axis("off")
    
    
    # Plot the training points
    for (q, j) in enumerate(np.unique(y_set)):
        plt.scatter(X_set[:, 0].reshape((len(X_set), 1))[y_set == j], X_set[:, 1].reshape((len(X_set), 1))[y_set == j], s = 20, c = ListedColormap(('red', 'green', 'blue'))(q), label = j)
        # plt.xlabel('Age')
        # plt.ylabel('Estimated Salary')
        plt.legend()
    
    # Accuracy on Test set
    # y_pred = clf.predict(pca_X_test)
    # acc_score = round(accuracy_score(y_test, y_pred), 2)
    
    # plt.title(f'{title[i]}, Acc(Test Set) = {acc_score}')
    # plt.tight_layout()
    
#%% Create a mesh to plot, Decision boundary (Test Set)
X_set, y_set = LDA_X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.5),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.5))


from matplotlib.colors import ListedColormap
# color_map = {-1: (1, 1, 1), 0: (0, 0, 0.9), 1: (1, 0, 0), 2: (0.8, 0.6, 0)}

fig = plt.figure(3)
fig.suptitle('Decision Boundary Comparison [Test Set]',fontweight ='bold')

# for i in range(0, len(classifiers)):
for (i, clf) in enumerate(classifiers):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.figure(2)
    plt.subplot(3, 3, i + 1)
    Z = clf.predict(np.array([X1.ravel(), X2.ravel()]).T)
    # Put the result into a color plot
    Z = Z.reshape(X1.shape)
    # plt.contourf(X1, X2, Z, cmap=plt.cm.Paired)
    plt.contourf(X1, X2, Z, alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    
    # plt.axis("off")
    
    
    # Plot the training points
    for (q, j) in enumerate(np.unique(y_set)):
        plt.scatter(X_set[:, 0].reshape((len(X_set), 1))[y_set == j], X_set[:, 1].reshape((len(X_set), 1))[y_set == j], s = 20, c = ListedColormap(('red', 'green', 'blue'))(q), label = j)
        # plt.xlabel('Age')
        # plt.ylabel('Estimated Salary')
        plt.legend()
    
    # Accuracy on Test set
    y_pred = clf.predict(LDA_X_test)
    acc_score = round(accuracy_score(y_test, y_pred), 2)
    
    plt.title(f'{title[i]}, Acc(Test Set) = {acc_score}')
    # plt.tight_layout()