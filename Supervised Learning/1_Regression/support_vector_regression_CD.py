# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:12:23 2022

@author: Chakron.D
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Settings
np.set_printoptions(precision=2)

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)
print(f'X = {X}')
print(f'y = {y}') 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
SS_X = StandardScaler()
SS_y = StandardScaler()

sc_X = SS_X.fit_transform(X) # new scaled X, X = SS_x.inverse_transform(sc_X)
sc_y = SS_y.fit_transform(y) # new scaled y, y = SS_y.inverse_transform(sc_y)

# Training the SVR
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(sc_X, sc_y) # regressor fit the scaled X and y

# Predict a new result
X_test = np.array([[6.5]])
sc_X_test = SS_X.transform(X_test)
SS_y.inverse_transform(regressor.predict(sc_X_test)) # inverse from scaled to real value

# Plot Support Vector Regression[SVR] Regression Result
plt.scatter(X, y, color='red') # X = SS_x.inverse_transform(sc_X), y = SS_y.inverse_transform(sc_y)
plt.plot(X, SS_y.inverse_transform(regressor.predict(sc_X)), color='blue')
plt.title('Truth or Bluff (Support Vector Regression[SVR])')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

# Plot Support Vector Regression[SVR] Result (Higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
sc_X_grid = SS_X.fit_transform(X_grid)
plt.figure(1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, SS_y.inverse_transform(regressor.predict(sc_X_grid)), color='blue')
plt.title('Truth or Bluff (Support Vector Regression - Higher resolution)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()


