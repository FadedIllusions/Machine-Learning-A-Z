#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Thompson Sampling
#
# Differs from UCB in that it constructs probability distributions of expected 
# values, rather than using the deterministic (strict) view of the UCB.
# Based on those distributions, it makes a selection and adjusts its beliefs
# accordingly, further refining the distribution of that selection.

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import DataSet
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
rounds = 10000
num_of_ads = 10
ads_selected = []
num_of_rewards_1 = [0] * num_of_ads
num_of_rewards_0 = [0] * num_of_ads
total_reward = 0
for rnd in range(0, rounds):
    ad = 0
    max_random = 0
    for version_of_ad in range(0, num_of_ads):
        random_beta = random.betavariate(num_of_rewards_1[version_of_ad] + 1, num_of_rewards_0[version_of_ad] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = version_of_ad
    ads_selected.append(ad)
    reward = dataset.values[rnd, ad]
    if reward == 1:
        num_of_rewards_1[ad] = num_of_rewards_1[ad] + 1
    else:
        num_of_rewards_0[ad] = num_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
best_ad = "Ad Number: " +str(ads_selected[rounds-1]+1)
#print(best_ad)

# Visualising the Results -- Histogram
plt.hist(ads_selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ads')
plt.ylabel('# Times Selected')          
plt.show()
