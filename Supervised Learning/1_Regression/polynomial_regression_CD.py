# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:23:25 2022

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
print(f'X = {X}')
print(f'y = {y}')

# Training the Linear Regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y) # train network

# Plot Linear Regression Result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

# Training the Polynomial Regressor model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Plot Polynomial Linear Regression Result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Linear Regression)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

# Plot Polynomial Linear Regression Result (Higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
X_poly_grid = poly_reg.fit_transform(X_grid)
plt.figure(2)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(X_poly_grid), color='blue')
plt.title('Truth or Bluff (Polynomial Linear Regression - Higher resolution)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()

# Predict a new result (Linear Regression)
X_test = np.array([[6.5, 5, 3]])
lin_reg.predict(X_test.T)

# Predict a new result (Linear Regression)
X_test = np.array([[6.5, 5, 3]])
X_poly_test = poly_reg.fit_transform(X_test.T)
lin_reg_2.predict(X_poly_test)





