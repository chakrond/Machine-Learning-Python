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
    
    # Update params
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