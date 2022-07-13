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
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)
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
# scaled_y_train = sc_y.fit_transform(y_train)

# Train LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(scaled_X_train, y_train)

# Make prediction
obv_test = np.array([[30, 87000]])
scaled_obv_test = sc_X.transform(obv_test)
print('Single Prediction of [30, 87000]', classifier.predict(scaled_obv_test))

scaled_X_test = sc_X.transform(X_test)
y_pred = classifier.predict(scaled_X_test)

# Plot
from mpl_toolkits.mplot3d import Axes3D
%matplotlib qt5
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, zdir='z')
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)

# Making confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
c_mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(classifier, scaled_X_test, y_test)
print('Confusion Matrix: ', c_mat)
# Check prediciton accuracy
accuracy_score(y_test, y_pred)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 50),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 50))

plt.contourf(X1, X2, classifier.predict(sc_X.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

enum_y_set = enumerate(np.unique(y_set))
print(list(enum_y_set))

for (i, j) in enumerate(np.unique(y_set)):
    plt.scatter(X_set[:, 0].reshape((len(X_set), 1))[y_set == j], X_set[:, 1].reshape((len(X_set), 1))[y_set == j], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 50),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 50))

plt.contourf(X1, X2, classifier.predict(sc_X.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for (i, j) in enumerate(np.unique(y_set)):
    plt.scatter(X_set[:, 0].reshape((len(X_set), 1))[y_set == j], X_set[:, 1].reshape((len(X_set), 1))[y_set == j], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


