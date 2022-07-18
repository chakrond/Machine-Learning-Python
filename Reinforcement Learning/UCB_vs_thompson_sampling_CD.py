# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 14:04:32 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
import random

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
plt.subplot(2, 2, 1)

plt.hist(ads_selected, bins=np.arange(-0.5, len(dataset.values[0])))
plt.title('Histogram of Selections - UCB')
# plt.xlabel('Advertisement No.')
plt.xticks(np.arange(0, len(dataset.values[0]), step=1))
# plt.ylabel('Number of Times')
plt.ylim([0, 10000])

# Reward Plot
plt.subplot(2, 2, 2)

# update value
y_ad = np.arange(len(dataset.values[0]))
plot_bar_rew = plt.barh(y_ad, sum_rewards)

# plot layout settings
plt.title('Reward Accumulation - UCB')

plt.xlim([0, 3000])
# plt.xlabel('Accumulated Rewards')

plt.yticks(y_ad, [f'Ad {i}' for i in y_ad])


#%% Thompson Sampling params
N               = 10000   # number of round
d               = len(dataset.values[0])  # number of ads
ads_selected    = []
n_rewards_1     = [0] * d
n_rewards_0     = [0] * d
sum_rewards     = [0] * d
total_rewards   = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    
    # for i in range(0, d):
    #     random_beta = random.betavariate(n_rewards_1[i] + 1, n_rewards_0[i] + 1)

    #     if (random_beta > max_random):
    #         max_random = random_beta
    #         ad = i
    
    
    random_beta = [random.betavariate(n_rewards_1[i] + 1, n_rewards_0[i] + 1) for i in range(0, d)]
    max_random  = max(random_beta)
    max_index   = np.array(random_beta).argmax()
    
    # # Update params
    ad          = max_index
    ads_selected.append(ad)
    reward      = dataset.values[n, ad]
    
    if reward == 1:
        n_rewards_1[ad] += 1
    else:
        n_rewards_0[ad] += 1
        
    # Update reward
    sum_rewards[ad] = sum_rewards[ad] + reward
    total_rewards   = total_rewards + reward


# Histogram
plt.figure(1)
plt.subplot(2, 2, 3)

plt.hist(ads_selected, bins=np.arange(-0.5, len(dataset.values[0])))
plt.title('Histogram of Selections - Thompson')
plt.xlabel('Advertisement No.')
plt.xticks(np.arange(0, len(dataset.values[0]), step=1))
plt.ylabel('Number of Times')
plt.ylim([0, 10000])

# Reward Plot
plt.subplot(2, 2, 4)

# update value
y_ad = np.arange(len(dataset.values[0]))
plot_bar_rew = plt.barh(y_ad, sum_rewards)

# plot layout settings
plt.title('Reward Accumulation - Thompson')

plt.xlim([0, 3000])
plt.xlabel('Accumulated Rewards')

plt.yticks(y_ad, [f'Ad {i}' for i in y_ad])

