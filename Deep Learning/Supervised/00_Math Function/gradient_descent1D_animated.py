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
fx_0 = 3*x**2 - 3*x + 4
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
learning_rate = .01
training_epochs = 150

#%% plot the results

# a range for x
coord_x = np.linspace(-2, 2, 2001)

# random starting point
localmin = np.random.choice(coord_x, 1)

# Observation List
n_step = 5
ob_list = np.arange(0, training_epochs + n_step, n_step)

# store the results
modelparams = np.zeros((training_epochs, 2))

for i in range(training_epochs):
    
    grad = deriv(localmin)
    localmin = localmin - learning_rate*grad
    modelparams[i, :] = localmin, grad
    
    if i == 0:
    
        # Figure Number
        fig = plt.figure(0)
        fig.suptitle(f'Gradient Descent 1D - iteration = {i}',fontweight ="bold")

        # Plot functions
        plt.subplot(2, 2, 1)
        plt.grid()
        plt.plot(coord_x, fx(coord_x), coord_x, deriv(coord_x))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend(['f(x)','df','f(x) min'])
        plt.title(f'Empirical local minimum: {localmin[0]:.5f}')
        
        # Plot localmin
        p_deriv_localmin = plt.plot(localmin, deriv(localmin),'ro')
        p_deriv_localmin.append(plt.text(localmin, deriv(localmin), s = f'({localmin[0]:.1f},{deriv(localmin)[0]:.1f})'))
        
        p_localmin = plt.plot(localmin, fx(localmin),'ro')
        p_localmin.append(plt.text(localmin, fx(localmin), s = f'({localmin[0]:.1f},{fx(localmin)[0]:.1f})'))

        # Plot Local minimum
        plt.subplot(2, 2, 2)
        plt.grid()
        plt.plot(modelparams[i, 0],'bo-')
        plt.xlabel('Iteration')
        plt.ylabel('Local minimum')
        plt.title(f'Estimated minimum: {localmin[0]:.5f}')
        
        
        # Plot Derivative Local minimum
        plt.subplot(2, 2, 3)
        plt.grid()
        plt.plot(modelparams[i, 1],'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Derivative')
        plt.title(f'Estimated minimum: {localmin[0]:.5f}')
        
        
    if sum(i == ob_list) > 0:
        
        # Figure name
        fig.suptitle(f'Gradient Descent 1D - iteration = {i}',fontweight ="bold")
        
        
        # Plot functions
        plt.subplot(2, 2, 1)
        plt.title(f'Empirical local minimum: {localmin[0]:.5f}')
        
        
        # Plot localmin
        p_deriv_localmin[0].set_xdata(localmin)
        p_deriv_localmin[0].set_ydata(deriv(localmin))
        p_deriv_localmin[1].set_x(localmin)
        p_deriv_localmin[1].set_y(deriv(localmin))
        p_deriv_localmin[1].set_text(f'({localmin[0]:.1f},{deriv(localmin)[0]:.1f})')
        
        p_localmin[0].set_xdata(localmin)
        p_localmin[0].set_ydata(fx(localmin))
        p_localmin[1].set_x(localmin)
        p_localmin[1].set_y(fx(localmin))
        p_localmin[1].set_text(f'({localmin[0]:.1f},{fx(localmin)[0]:.1f})')
        
        # Plot Local minimum
        plt.subplot(2, 2, 2)
        plt.plot(i, modelparams[i, 0],'bo-')
        plt.title(f'Estimated minimum: {localmin[0]:.5f}')
    
    
        # Plot Derivative Local minimum
        plt.subplot(2, 2, 3)
        plt.plot(i, modelparams[i, 1],'bo-')
        plt.title(f'Derivative: {modelparams[i, 1]:.5f}')
    
    
        plt.draw()
        
        
    time.sleep(0.01)
    plt.pause(0.01)



