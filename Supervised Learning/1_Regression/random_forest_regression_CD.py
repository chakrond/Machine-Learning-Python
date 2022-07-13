# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 22:51:44 2022

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

# Training the Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

# Prediction a new result
X_test = np.array([[6.5, 1.1]])
# X_test = X_test.reshape(len(X_test), 1)
regressor.predict(X_test.T)

# Plot Random Forest Regression Result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.figure(0)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Random Forest Regression - Higher resolution)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()