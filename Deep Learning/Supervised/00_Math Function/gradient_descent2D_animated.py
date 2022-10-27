# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:35:14 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
sx, sy = sym.symbols('sx, sy')

# peaks function
sZ = 3*(1-sx)**2 * sym.exp(-(sx**2) - (sy+1)**2) \
      - 10*(sx/5 - sx**3 - sy**5) * sym.exp(-sx**2-sy**2) \
      - 1/3*sym.exp(-(sx+1)**2 - sy**2)


# sympy derivatives functions
fsZ = sym.lambdify( (sx, sy), sZ, 'numpy')
df_x = sym.lambdify( (sx, sy), sym.diff(sZ, sx), 'sympy' )
df_y = sym.lambdify( (sx, sy), sym.diff(sZ, sy), 'sympy' )

# df_x(1, 1).evalf()
    
#%% parameters

# learning parameters
learning_rate = 0.01
training_epochs = 1000

#%% plot the results

# random starting point
# local_min = np.random.rand(2)
local_min = np.array([-0.55, -0.55])
# local_min = np.array([-0.3, -0.7])

# generate coordinates
coord_x = np.linspace(-3, 3, 201)
coord_y = np.linspace(-3, 3, 201)
coord_z = np.zeros( (len(coord_x), len(coord_y)) )

# compute coord_z
for i in range(len(coord_y)):
    coord_z[i, 0] = fsZ(coord_x[0], coord_y[i])
    
    for k in range(len(coord_x)):
        coord_z[i, k] = fsZ(coord_x[k], coord_y[i])

ext = coord_x[0], coord_x[-1], coord_y[0], coord_y[-1]

# Observation List
n_step = 5
ob_list = np.arange(0, training_epochs + n_step, n_step)

# store the results
model_local_min = np.zeros((training_epochs, 2))
model_grad = np.zeros((training_epochs, 2))

for i in range(training_epochs):
    grad = np.array([ df_x(local_min[0], local_min[1]).evalf(), 
                      df_y(local_min[0], local_min[1]).evalf() 
                    ])
    
    # normal learning rate
    lr = learning_rate
    
    # scaled by abs gradient get dynamic learning rate
    # lr = learning_rate*abs(grad)  # based on value of gradient(derivative) - RMSprop, Adam Optimizer
    # lr = learning_rate*(1 - (i+1)/training_epochs) # based on iteration(time-based) - Learning rate decay
    
    local_min = local_min - lr*grad
    model_local_min[i, :] = local_min
    model_grad[i, :] = grad
  
    if i == 0:
    
        # Figure Number
        fig = plt.figure(0)
        fig.suptitle(f'Gradient Descent 2D - iteration = {i}', fontweight ="bold")

        # Plot area
        # plt.subplot(2, 2, 1)
        
        plt.imshow(coord_z, extent=ext, vmin=-5,vmax=5, origin='lower')
        plt.colorbar()
        plt.show()
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.legend(['f(x)','df','f(x) min'])
        plt.title(f'Empirical local minimum: {fsZ(model_local_min[i][0], model_local_min[i][1]):.3f}')
        # plt.xlim([-2, 2])
        # plt.ylim([-20, 20])
        
        # Start point
        plt.plot(model_local_min[i][0], model_local_min[i][1],'bo') # start point
        plt.text(model_local_min[i][0], model_local_min[i][1],
                                          s = f'({model_local_min[i][0]:.2f},{model_local_min[i][1]:.2f},{fsZ(model_local_min[i][0], model_local_min[i][1]):.2f})')
        
        
        # Plot local_min
        p_local_min = plt.plot(model_local_min[i][0], model_local_min[i][1],'ro')
        p_local_min.append(plt.text(model_local_min[i][0], model_local_min[i][1],
                                          s = f'({model_local_min[i][0]:.2f},{model_local_min[i][1]:.2f},{fsZ(model_local_min[i][0], model_local_min[i][1]):.2f})'))
        
        plt.legend(['start point','local min'])

    if sum(i == ob_list) > 0:
        
        # Figure name
        fig.suptitle(f'Gradient Descent 2D - iteration = {i}',fontweight ="bold")
        
        
        # Plot functions
        plt.title(f'Empirical local minimum: {fsZ(model_local_min[i][0], model_local_min[i][1]):.3f}')
        
        
        # Plot local min
        p_local_min[0].set_xdata(model_local_min[i][0])
        p_local_min[0].set_ydata(model_local_min[i][1])
        p_local_min[1].set_x(model_local_min[i][0])
        p_local_min[1].set_y(model_local_min[i][1])
        p_local_min[1].set_text(f'({model_local_min[i][0]:.2f},{model_local_min[i][1]:.2f},{fsZ(model_local_min[i][0], model_local_min[i][1]):.2f})')
     
        # add track point
        plt.plot(model_local_min[i][0], model_local_min[i][1],'m.', alpha=0.3)
        
        # update the plot
        plt.draw()
        
    time.sleep(0.01)
    plt.pause(0.01)
        
#%% 3D surface plot

X, Y = np.meshgrid(coord_x, coord_y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, coord_z, cmap=cm.jet, linewidth=0, antialiased=False)
ax.scatter(coord_x[95], coord_y[154], fsZ(coord_x[95], coord_y[154]), s=200, marker='o', c='m')
plt.draw()


#%% 3D surface plot of Derivatives

# df_coord_z = np.zeros((len(model_grad), len(model_grad)))

# compute coord_z
# for i in range(len(model_grad)):
#     df_coord_z[i, 0] = fsZ(model_grad[i, 0], model_grad[i, 1])
    
#     for k in range(len(model_grad)):
#         df_coord_z[i, k] = fsZ(model_grad[k, 0], model_grad[i, 1])

# plt.imshow(df_coord_z, extent=ext, origin='lower')

# X, Y = np.meshgrid(model_grad[:, 0], model_grad[:, 1])

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(X, Y, df_coord_z, cmap=cm.jet, linewidth=0, antialiased=False)
# ax.scatter(coord_x[95], coord_y[154], fsZ(coord_x[95], coord_y[154]), s=200, marker='o', c='m')
# plt.draw()