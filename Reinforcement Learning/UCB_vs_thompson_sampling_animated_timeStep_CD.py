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

#%% UCB vs Thompson Animation

N               = 10000   # number of round
d               = 10      # number of ads

# UCB params
ads_selected_UCB    = []
n_selections_UCB    = [0] * d
sum_rewards_UCB     = [0] * d
total_reward_UCB    = 0

# Thompson params
ads_selected_TS     = []
n_selections_TS     = [0] * d
n_rewards_1         = [0] * d
n_rewards_0         = [0] * d
sum_rewards_TS      = [0] * d
total_rewards_TS    = 0

# Observation List
ob_list = np.arange(100, N, 100)


for n in range(0, N):
    
    # --------------------------------
    # UCB
    # --------------------------------
    ad_UCB = 0
    max_upper_bound = 0
    
    # --------------------------------
    # TS
    # --------------------------------
    ad_TS = 0
    max_random = 0
    
    # --------------------------------
    # UCB
    # --------------------------------
    for i in range(0, d):
        
        if (n_selections_UCB[i] > 0):
            avg_reward  = sum_rewards_UCB[i] / n_selections_UCB[i]
            delta_i     = math.sqrt( (3/2) * (math.log(n+1) / n_selections_UCB[i]) )
            upper_bound = avg_reward + delta_i   
            
        else: 
            upper_bound = 1e400 # super high value (infinity)
            
        if (upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad_UCB = i
            
    # update the list
    ads_selected_UCB.append(ad_UCB)
    n_selections_UCB[ad_UCB]    += 1
    reward_UCB                  = dataset.values[n, ad_UCB]
    sum_rewards_UCB[ad_UCB]     = sum_rewards_UCB[ad_UCB] + reward_UCB
    total_reward_UCB            = total_reward_UCB + reward_UCB
    
    
    # --------------------------------
    # TS
    # --------------------------------
    random_beta = [random.betavariate(n_rewards_1[i] + 1, n_rewards_0[i] + 1) for i in range(0, d)]
    max_random  = max(random_beta)
    max_index   = np.array(random_beta).argmax()
    
    # # Update params
    ad_TS                   = max_index
    n_selections_TS[ad_TS] += 1
    ads_selected_TS.append(ad_TS)
    reward_TS               = dataset.values[n, ad_TS]
    
    if reward_TS == 1:
        n_rewards_1[ad_TS] += 1
    else:
        n_rewards_0[ad_TS] += 1
        
    # Update reward
    sum_rewards_TS[ad_TS] = sum_rewards_TS[ad_TS] + reward_TS
    total_rewards_TS   = total_rewards_TS + reward_TS
    
    
    
    if (n == 0):
        
        # Figure number
        fig = plt.figure(2)
        fig.suptitle(f'UCB vs Thompson - iteration = {n}',fontweight ="bold")
        
        # --------------------------------
        # UCB
        # --------------------------------
        
        # Selection History Plot
        plt.subplot(2, 2, 1)
        
        # update value
        y_ad = np.arange(len(dataset.values[0]))
        plot_bar_UCB = plt.barh(y_ad, 0)
        plot_bar_UCB[ad_UCB].set_width(1)
        
        # plot layout settings
        plt.title('Selection History - UCB')
        
        plt.xlim([0, 5])
        # plt.xlabel('Number of Times')
        
        plt.yticks(y_ad, [f'Ad {i}' for i in y_ad])
        # plt.ylabel('Advertisement No.')
        
        # plt.draw()
        
        
        # Reward Plot
        plt.subplot(2, 2, 2)
        
        # update value
        y_ad = np.arange(len(dataset.values[0]))
        plot_bar_rew_UCB = plt.barh(y_ad, 0)
        plot_bar_rew_UCB[ad_UCB].set_width(plot_bar_rew_UCB[ad_UCB].get_width() + reward_UCB)
        
        # plot layout settings
        plt.title('Reward Accumulation - UCB')
        
        plt.xlim([0, 5])
        # plt.xlabel('Accumulated Rewards')
        
        plt.yticks(y_ad, [f'Ad {i}' for i in y_ad])
        # plt.ylabel('Advertisement No.')
        
        # plt.draw()
        
        # --------------------------------
        # TS
        # --------------------------------
        
        # Selection History Plot
        plt.subplot(2, 2, 3)
        
        # update value
        y_ad = np.arange(len(dataset.values[0]))
        plot_bar_TS = plt.barh(y_ad, 0)
        plot_bar_TS[ad_TS].set_width(1)
        
        # plot layout settings
        plt.title('Selection History - Thompson')
        
        plt.xlim([0, 5])
        plt.xlabel('Number of Times')
        
        plt.yticks(y_ad, [f'Ad {i}' for i in y_ad])
        plt.ylabel('Advertisement No.')
        
        # plt.draw()
        
        
        # Reward Plot
        plt.subplot(2, 2, 4)
        
        # update value
        y_ad = np.arange(len(dataset.values[0]))
        plot_bar_rew_TS = plt.barh(y_ad, 0)
        plot_bar_rew_TS[ad_TS].set_width(plot_bar_rew_TS[ad_TS].get_width() + reward_TS)
        
        # plot layout settings
        plt.title('Reward Accumulation - Thompson')
        
        plt.xlim([0, 5])
        plt.xlabel('Accumulated Rewards')
        
        plt.yticks(y_ad, [f'Ad {i}' for i in y_ad])
        
        
    if sum(n == ob_list) > 0 :
        
        fig.suptitle(f'UCB vs Thompson - iteration = {n}',fontweight ="bold")
        
        
        # X, Y axis update
        
        # Histrogram Bar
        max_X_lim_bar  = max(n_selections_UCB + n_selections_TS)
        offset_X_bar   = 1.1
        
        # Reward Bar
        max_X_lim_bar_rew  = max(sum_rewards_UCB + n_rewards_1)
        offset_X_bar_rew   = 1.1
        
        
        # --------------------------------
        # UCB
        # --------------------------------

        # Selection History Plot
        plt.subplot(2, 2, 1)
        plt.xlim([0, max_X_lim_bar*offset_X_bar])
        
        for (i, bar) in enumerate(plot_bar_UCB):
            bar.set_width(n_selections_UCB[i]) 
    
        
        # Reward Plot
        plt.subplot(2, 2, 2)
        plt.xlim([0, max_X_lim_bar_rew*offset_X_bar_rew])
        
        for (i, bar) in enumerate(plot_bar_rew_UCB):
            bar.set_width(sum_rewards_UCB[i]) 
        
        # --------------------------------
        # TS
        # --------------------------------

        # Selection History Plot
        plt.subplot(2, 2, 3)
        plt.xlim([0, max_X_lim_bar*offset_X_bar])
        
        for (i, bar) in enumerate(plot_bar_TS):
            bar.set_width(n_selections_TS[i])
        
        # Reward Plot
        plt.subplot(2, 2, 4)
        plt.xlim([0, max_X_lim_bar_rew*offset_X_bar_rew])
        
        for (i, bar) in enumerate(plot_bar_rew_TS):
            bar.set_width(n_rewards_1[i]) 


        
    
    time.sleep(0.0001)
    plt.pause(0.0001)

