# -*- coding: utf-8 -*-

# Decision Tree Regression
# (A Non-Continuous Regression)

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import DataSet
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting DTR To Dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

# Predict A New Result
y_pred = regressor.predict(6.5)

# Visualize DTR Results
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Data Tree Regression")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()

# Visualize DTR Results (With Higher Resolution)
# Note that DTR is based upon an average of dependent variable values;
# Thus, it tends to work with ranges/steps
# DTR tends to be of better use when working with dimensions above 1D
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Data Tree Regression (HR)")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()
