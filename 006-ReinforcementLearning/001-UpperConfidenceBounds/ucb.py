#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Upper Confidence Bounds

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


## --- -- - Implement UCB - -- --- ##
import math
rounds = 10000
num_of_ads = 10
ads_selected = []
num_of_selections = [0]*num_of_ads
sum_of_rewards = [0]*num_of_ads
total_rewards = 0

for rnd in range(0, rounds):
	ad = 0
	max_upper_bound = 0
	for version_of_ad in range(0, num_of_ads):
		if (num_of_selections[version_of_ad] > 0):
			average_reward = sum_of_rewards[version_of_ad]/num_of_selections[version_of_ad]
			delta_version_of_ad = math.sqrt(3/2*math.log(rnd+1)/num_of_selections[version_of_ad])
			upper_bound = average_reward + delta_version_of_ad
		else:
			upper_bound = 1e400
		if upper_bound > max_upper_bound:
			max_upper_bound = upper_bound
			ad = version_of_ad

	ads_selected.append(ad)
	num_of_selections[ad] += 1
	reward = dataset.values[rnd, ad]
	sum_of_rewards[ad] += reward
	total_rewards += reward

best_ad = "Ad Number: " +str(ads_selected[rounds-1]+1)
#print(best_ad)

# Visualising the Results -- Histogram
plt.hist(ads_selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ads')
plt.ylabel('# Times Selected')          
plt.show()
