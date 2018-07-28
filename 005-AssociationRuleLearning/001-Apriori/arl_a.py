#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Analytics: Beer and Diapers...
# www.theregister.co.uk/2006/08/15/beer_diapers/

# Association Rule - Apriori
# Apriori analyzes things that come in pairs. Id est, in the aforementioned
# example, those who bought diapers had a higher likelihood of buying beer.
# 
# 3 Parts of Aprior: Support, Confidence, Lift
# (See Associated PNGs for diagramatical intuition.)
#
# Support (Similar to Bayes'): -- Using Movies In Example
# support(x) = # Watched Lists Containing M / # Total Watched Lists
# 
# Confidence:
# confidence(M1->M2) = # Lists Containing M1 & M2 / # Lists Containing M1
#
# Lift:
# lift(M1->M2) = confidence(M1->M2)/support(M2)

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import Dataset
# 'header = none' option specifies that the columns in the dataset do
# not have a title; therefor, items are placed in a column and not as
# titles of the dataset columns.
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for transaction in range(0,dataset.shape[0]):
	transactions.append([str(dataset.values[transaction,column]) for column in range(0,dataset.shape[1])])

# Training Apriori on the Dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# In min_support, we wanted products bought at least 3 times a day (7 days/week). 
# So, we have 3*7 / total products (7500). 0.0028. And, we rounded up.
# min_confidence is a bit more trial and error.
# min_lift is, also, a bit more trial and error.
# min_length is 2 since we don't want associations containing on a single item.

# Visualizing the Results
results = list(rules)
# Open 'value' field of 'results' variable to see associations.
