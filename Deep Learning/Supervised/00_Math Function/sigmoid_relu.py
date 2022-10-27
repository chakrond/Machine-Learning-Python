# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 01:27:43 2022

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

# symbolic math in Python
import sympy as sym
import sympy.plotting.plot as symplot

cd = os.getcwd()
# os.chdir(cd) # set the working directory

# Settings
np.set_printoptions(precision=4)

#%% relu and sigmoid

x = sym.symbols('x')

# symbolic functions
relu = sym.Max(0,x)
sigmoid = 1 / (1+sym.exp(-x))

# graph functions
p1 = symplot(relu,(x,-4,4),label='ReLU',show=False,line_color='blue')
p1.extend( symplot(sigmoid,(x,-4,4),label='Sigmoid',show=False,line_color='red') )
p1.legend = True
p1.title = 'The functions'
p1.show()

# graph derivatives of funcitons
p2 = symplot(sym.diff(relu),(x,-4,4),label='df(ReLU)',show=False,line_color='blue')
p2.extend( symplot(sym.diff(sigmoid),(x,-4,4),label='df(Sigmoid)',show=False,line_color='red') )
p2.legend = True
p2.title = 'The derivatives'
p2.show()