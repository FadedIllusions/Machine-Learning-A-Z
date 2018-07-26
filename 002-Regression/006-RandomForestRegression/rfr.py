# -*- coding: utf-8 -*-

# Random Forest Regression
# (RFR is a 'team' of several Decision Trees)
# (Prediction of RFR is an average of the decision trees)

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import DataSet
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting RFR to DataSet
# Estimators = Number of Decision Trees in Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,y)

# Predicting a New Value
y_pred = regressor.predict(6.5)

# Visualizing RFR Results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Data Tree Regression (HR)")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()
