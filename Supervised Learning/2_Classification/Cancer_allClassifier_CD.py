# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 14:24:36 2022

@author: Chakron.D
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Settings
np.set_printoptions(precision=2)

# Importing the dataset
dataset = pd.read_csv('Data_Cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# y = y.reshape(len(y), 1)
print(f'X = {X}')
print(f'y = {y}') 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler() # if y is already between 0 and 1, no need to scale
scaled_X_train = sc_X.fit_transform(X_train)
scaled_X_test = sc_X.transform(X_test)
# scaled_y_train = sc_y.fit_transform(y_train.reshape(-1, 1))

# Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


X_train_set, y_train_set = scaled_X_train, y_train

# test = LogisticRegression(random_state=0)
# test.fit(X_train_set, np.ravel(y_train_set))

c1 = (LogisticRegression(random_state=0).fit(X_train_set, y_train_set), "Logistic Regression")
c2 = (KNeighborsClassifier(n_neighbors=5, metric='minkowski').fit(X_train_set, y_train_set), "KNeighbors Classifier")
c3 = (SVC(kernel = 'linear', random_state=0).fit(X_train_set, y_train_set), "SVC with Linear kernel")
c4 = (SVC(kernel = 'rbf', random_state=0).fit(X_train_set, y_train_set), "SVC with RBF kernel")
c5 = (GaussianNB().fit(X_train_set, y_train_set), "Gaussian Naive Bayes")
c6 = (DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train_set, y_train_set), "Decision Tree")
c7 = (RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0).fit(X_train_set, y_train_set), "Random Forest")

classifiers = (c1, c2, c3, c4, c5, c6, c7)


# Making confusion matrix   
from sklearn.metrics import accuracy_score, plot_confusion_matrix

nrows = 3
ncols = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(15,10))

for i in range(0, nrows*ncols):
    axes.flatten()[i].axis("off")
 
for i, (clf, title) in enumerate(classifiers):
    
    fig.add_subplot(axes.flatten()[i])
    plot_confusion_matrix(clf, scaled_X_test, y_test, ax=axes.flatten()[i], colorbar=False)
    y_pred = clf.predict(scaled_X_test)
    acc_score = round(accuracy_score(y_test, y_pred), 2)
    axes.flatten()[i].title.set_text(f'{title}, Accuracy Score = {acc_score}')
    plt.axis("on")
    plt.tight_layout()

