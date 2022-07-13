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
# sc_y = StandardScaler() # y is already between 0 and 1, no need to scale
scaled_X_train = sc_X.fit_transform(X_train)
scaled_X_test = sc_X.fit_transform(X_test)
scaled_y_train = sc_y.fit_transform(y_train)

# Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


X_train_set, y_train_set = scaled_X_train, scaled_y_train

c1 = (LogisticRegression(random_state=0).fit(X_train_set, y_train_set), "Logistic Regression")
c2 = (KNeighborsClassifier(n_neighbors=5, metric='minkowski').fit(X_train_set, y_train_set), "KNeighbors Classifier")
c3 = (SVC(kernel = 'linear', random_state=0).fit(X_train_set, y_train_set), "SVC with Linear kernel")
c4 = (SVC(kernel = 'rbf', random_state=0).fit(X_train_set, y_train_set), "SVC with RBF kernel")
c5 = (GaussianNB().fit(X_train_set, y_train_set), "GaussianNB")
c6 = (DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train_set, y_train_set), "Decision Tree")
c7 = (RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0).fit(X_train_set, y_train_set), "Random Forest")

classifiers = (c1, c2, c3, c4, c5, c6, c7)


# create a mesh to plot
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 2),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 2))


from matplotlib.colors import ListedColormap
color_map = {-1: (1, 1, 1), 0: (0, 0, 0.9), 1: (1, 0, 0), 2: (0.8, 0.6, 0)}


# for i in range(0, len(classifiers)):
for i, (clf, title) in enumerate(classifiers):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.figure(0)
    plt.subplot(3, 3, i + 1)
    Z = clf.predict(sc_X.transform(np.array([X1.ravel(), X2.ravel()]).T))
    # Put the result into a color plot
    Z = Z.reshape(X1.shape)
    # plt.contourf(X1, X2, Z, cmap=plt.cm.Paired)
    plt.contourf(X1, X2, Z, alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    
    # plt.axis("off")
    
    
    # Plot the training points
    for (q, j) in enumerate(np.unique(y_set)):
        plt.scatter(X_set[:, 0].reshape((len(X_set), 1))[y_set == j], X_set[:, 1].reshape((len(X_set), 1))[y_set == j], s = 1.5, c = ListedColormap(('red', 'green'))(q), label = j)
        # plt.xlabel('Age')
        # plt.ylabel('Estimated Salary')
        plt.legend()
        
    plt.title(title)

# Making confusion matrix   
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix

nrows = 3
ncols = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(15,10))

for i in range(0, nrows*ncols):
    axes.flatten()[i].axis("off")
 
for i, (clf, title) in enumerate(classifiers):
    
    fig.add_subplot(axes.flatten()[i])
    plot_confusion_matrix(clf, scaled_X_test, y_test, ax=axes.flatten()[i], colorbar=False)
    y_pred = clf.predict(scaled_X_test)
    acc_score = accuracy_score(y_test, y_pred)
    axes.flatten()[i].title.set_text(f'{title}, Accuracy Score = {acc_score}')
    plt.axis("on")
    plt.tight_layout()

