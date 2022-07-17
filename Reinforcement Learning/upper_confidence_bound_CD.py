# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 15:09:06 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time

# Settings
np.set_printoptions(precision=2)

#%% Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv') # transaction for one week

#%% UCB Parameters Init
N = 10000   # number of round
d = len(dataset.values[0])  # number of ads
ads_selected = []
n_selections = [0] * d
sum_rewards  = [0] * d
total_reward = 0
upper_bound  = np.zeros((N, d))

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    
    for i in range(0, d):
        
        if (n_selections[i] > 0):
            avg_reward  = sum_rewards[i] / n_selections[i]
            delta_i     = math.sqrt( (3/2) * (math.log(n+1) / n_selections[i]) )
            upper_bound[n, i] = avg_reward + delta_i   
            
        else: 
            upper_bound[n, i] = 1e400 # super high value (infinity)
            
        if (upper_bound[n, i] > max_upper_bound):
            max_upper_bound = upper_bound[n, i]
            ad = i
            
    # update the list
    ads_selected.append(ad)
    n_selections[ad] += 1
    reward            = dataset.values[n, ad]
    sum_rewards[ad]   = sum_rewards[ad] + reward
    total_reward      = total_reward + reward

# Histogram
plt.figure(1)
plt.subplot(1, 2, 1)

plt.hist(ads_selected, bins=np.arange(-0.5, len(dataset.values[0])))
plt.title('Histogram of Selections')
plt.xlabel('Advertisement No.')
plt.xticks(np.arange(0, len(dataset.values[0]), step=1))
plt.ylabel('Number of Times')
plt.show()

# Reward Plot
plt.subplot(1, 2, 2)

# update value
y_ad = np.arange(len(dataset.values[0]))
plot_bar_rew = plt.barh(y_ad, sum_rewards)

# plot layout settings
plt.title('Reward Accumulation')

plt.xlim([0, max(sum_rewards) * 1.1])
plt.xlabel('Accumulated Rewards')

plt.yticks(y_ad, [f'Ad {i}' for i in y_ad])


    

#%% UCB Animation
N = 10000   # number of round
d = 10      # number of ads
ads_selected = []
n_selections = [0] * d
sum_rewards  = [0] * d
total_reward = 0

for n in range(0, N):
    
    ad = 0
    max_upper_bound = 0
    
    for i in range(0, d):
        
        if (n_selections[i] > 0):
            avg_reward  = sum_rewards[i] / n_selections[i]
            delta_i     = math.sqrt( (3/2) * (math.log(n+1) / n_selections[i]) )
            upper_bound = avg_reward + delta_i   
            
        else: 
            upper_bound = 1e400 # super high value (infinity)
            
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
            
    # update the list
    ads_selected.append(ad)
    n_selections[ad] += 1
    reward            = dataset.values[n, ad]
    sum_rewards[ad]   = sum_rewards[ad] + reward
    total_reward      = total_reward + reward
    
    
    if (n == 0):
        
        # Figure number
        plt.figure(1)
        
        # Selection History Plot
        plt.subplot(1, 2, 1)
        
        # update value
        y_ad = np.arange(len(dataset.values[0]))
        plot_bar = plt.barh(y_ad, 0)
        plot_bar[ad].set_width(1)
        
        # plot layout settings
        plt.title('Selection History')
        
        plt.xlim([0, 5])
        plt.xlabel('Number of Times')
        
        plt.yticks(y_ad, [f'Ad {i}' for i in y_ad])
        plt.ylabel('Advertisement No.')
        
        plt.draw()
        
        
        # Reward Plot
        plt.subplot(1, 2, 2)
        
        # update value
        y_ad = np.arange(len(dataset.values[0]))
        plot_bar_rew = plt.barh(y_ad, 0)
        plot_bar_rew[ad].set_width(plot_bar_rew[ad].get_width() + reward)
        
        # plot layout settings
        plt.title('Reward Accumulation')
        
        plt.xlim([0, 5])
        plt.xlabel('Accumulated Rewards')
        
        plt.yticks(y_ad, [f'Ad {i}' for i in y_ad])
        # plt.ylabel('Advertisement No.')
        
        plt.draw()
        
    else:
        # Selection History Plot
        plt.subplot(1, 2, 1)
        max_lim  = max([bar.get_width() for bar in plot_bar])
        offset_X = 0.1 * max_lim
        plt.xlim([0, max_lim + offset_X])
        plot_bar[ad].set_width(plot_bar[ad].get_width() + 1)
        
        plt.draw()
        
        
        # Reward Plot
        plt.subplot(1, 2, 2)
        max_lim  = max([bar.get_width() for bar in plot_bar_rew])
        offset_X = 0.1 * max_lim
        plt.xlim([0, max_lim + offset_X])
        plot_bar_rew[ad].set_width(plot_bar_rew[ad].get_width() + reward)
        
        plt.draw()


        
    
    time.sleep(0.0001)
    plt.pause(0.0001)
