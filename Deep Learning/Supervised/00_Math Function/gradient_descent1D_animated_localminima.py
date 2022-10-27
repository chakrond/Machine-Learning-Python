# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 04:26:17 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import random
import os
import sympy as sym

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)
plt.rcParams['font.size'] = 10

#%% Gradient descent in 1D

# Create symbolic variables
x = sym.symbols('x')

# Create a function
# fx_0 = 3*x**2 - 3*x + 4
fx_0 = sym.cos(2*sym.pi*x) + x**2
fx_v = sym.lambdify(x, fx_0)

# compute derivative
df_0 = sym.diff(fx_0, x)
df_v = sym.lambdify(x, df_0)

# Function
def fx(v):
    return fx_v(v)

# Derivative function
def deriv(v):
    return df_v(v)
    
#%% parameters

# learning parameters
learning_rate = 0.01
training_epochs = 1000

#%% plot the results

# a range for x
coord_x = np.linspace(-2, 2, 2001)

# random starting point
local_min = np.random.choice(coord_x, 1)
# local_min = np.array([0]) # vanishing gradient

# Observation List
n_step = 5
ob_list = np.arange(0, training_epochs + n_step, n_step)

# store the results
model_params = np.zeros((training_epochs, 2))

for i in range(training_epochs):
    
    grad = deriv(local_min) # vanishing gradient if learning rate is too small and the model doesn't learn
    local_min = local_min - learning_rate*grad
    model_params[i, :] = local_min, grad
    
    if i == 0:
    
        # Figure Number
        fig = plt.figure(0)
        fig.suptitle(f'Gradient Descent 1D - iteration = {i}', fontweight ="bold")

        # Plot functions
        plt.subplot(2, 2, 1)
        plt.grid()
        plt.plot(coord_x, fx(coord_x), coord_x, deriv(coord_x))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend(['f(x)','df','f(x) min'])
        plt.title(f'Empirical local minimum: {local_min[0]:.5f}')
        plt.xlim([-2, 2])
        # plt.ylim([-20, 20])
        
        # Plot local_min
        p_deriv_local_min = plt.plot(local_min, deriv(local_min),'ro')
        p_deriv_local_min.append(plt.text(local_min, deriv(local_min), s = f'({local_min[0]:.1f},{deriv(local_min)[0]:.1f})'))
        
        p_local_min = plt.plot(local_min, fx(local_min),'ro')
        p_local_min.append(plt.text(local_min, fx(local_min), s = f'({local_min[0]:.1f},{fx(local_min)[0]:.1f})'))

        # Plot Local minimum
        plt.subplot(2, 2, 2)
        plt.grid()
        plt.plot(model_params[i, 0],'bo-')
        plt.xlabel('Iteration')
        plt.ylabel('Local minimum')
        plt.title(f'Estimated minimum: {local_min[0]:.5f}')
        
        
        # Plot Derivative Local minimum
        plt.subplot(2, 2, 3)
        plt.grid()
        plt.plot(model_params[i, 1],'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Derivative')
        plt.title(f'Estimated minimum: {local_min[0]:.5f}')
        
        
    if sum(i == ob_list) > 0:
        
        # Figure name
        fig.suptitle(f'Gradient Descent 1D - iteration = {i}',fontweight ="bold")
        
        
        # Plot functions
        plt.subplot(2, 2, 1)
        plt.title(f'Empirical local minimum: {local_min[0]:.5f}')
        
        
        # Plot local min
        p_deriv_local_min[0].set_xdata(local_min)
        p_deriv_local_min[0].set_ydata(deriv(local_min))
        p_deriv_local_min[1].set_x(local_min)
        p_deriv_local_min[1].set_y(deriv(local_min))
        p_deriv_local_min[1].set_text(f'({local_min[0]:.1f},{deriv(local_min)[0]:.1f})')
        
        p_local_min[0].set_xdata(local_min)
        p_local_min[0].set_ydata(fx(local_min))
        p_local_min[1].set_x(local_min)
        p_local_min[1].set_y(fx(local_min))
        p_local_min[1].set_text(f'({local_min[0]:.1f},{fx(local_min)[0]:.1f})')
        
        # Plot Local minimum
        plt.subplot(2, 2, 2)
        plt.plot(i, model_params[i, 0],'bo-')
        plt.title(f'Estimated minimum: {local_min[0]:.5f}')
    
    
        # Plot Derivative Local minimum
        plt.subplot(2, 2, 3)
        plt.plot(i, model_params[i, 1],'bo-')
        plt.title(f'Derivative: {model_params[i, 1]:.5f}')
    
    
        plt.draw()
        
        
    time.sleep(0.01)
    plt.pause(0.01)



