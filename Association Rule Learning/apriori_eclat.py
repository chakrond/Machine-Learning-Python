# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:44:59 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Settings
np.set_printoptions(precision=2)

#%% Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None) # transaction for one week

#%% Data processing
transaction = []
for i in range(0, len(dataset)):
    # list_ = []
    # for j in range(0, len(dataset.values[0])):
    #     list_.append(str(dataset.values[i, j]))
    # transaction.append(list_)
    transaction.append( [ str(dataset.values[i, j]) for j in range(0, len(dataset.values[0])) ] )

#%% Train the Apriori model 
from apyori import apriori
param_min_supp   = round( (3 * 7) / len(dataset), 3) # 3 times per day, all transaction were recorded for one week time
param_min_confd  = 0.2 # recommended min = 0.2
param_min_lift   = 3 # recommended min = 3
param_min_length = 2 # minimum of the final rules (buy one and get one free, min length = 2)
param_max_length = 2 # maximum of the final rules (buy one and get one free, max length = 2)

rules = apriori(transactions=transaction, min_support=param_min_supp, min_confidence=param_min_confd,
                min_lift=param_min_lift, min_length=param_min_length, max_length=param_max_length)

#%% Results

results = list(rules)

results[0][1] # support
tuple(results[0][2][0][0])[0] # first
tuple(results[0][2][0][1])[0] # second
results[0][2][0][2] # confidence
results[0][2][0][3] # lift

firstRule   = [tuple(i[2][0][0])[0] for i in results]
secondRule  = [tuple(i[2][0][1])[0] for i in results]
supports    = [i[1] for i in results]
confidences = [i[2][0][2] for i in results] # Eclat doesn't include confidence and lifts
lifts       = [i[2][0][3] for i in results] # Eclat doesn't include confidence and lifts

results_list_Apriori = list(zip(firstRule, secondRule, supports, confidences, lifts))
DataFrame_result_Apriori = pd.DataFrame(results_list_Apriori, columns = ['1st Rule', '2nd Rule', 'Support', 'Confidence', 'Lift'])

results_list_Eclat = list(zip(firstRule, secondRule, supports))
DataFrame_result_Eclat = pd.DataFrame(results_list_Eclat, columns = ['1st Rule', '2nd Rule', 'Support'])

