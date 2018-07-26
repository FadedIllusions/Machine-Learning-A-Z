# -*- coding: utf-8 -*-

# Polynomial Regression

# Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Dataset too small, not any need for train/test split

# Fitting LR to Dataset -- For Comparison
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting PR to dataset
from sklearn.preprocessing import PolynomialFeatures
# Adjust degree as needed, to better fit model
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualizing th LR Results
plt.scatter(X,y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Linear Regression Results")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Visualize the PR Results
# X_grid added to smooth graph by adding steps/increments
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title("Polynomial Regression Results")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Predict a new result with LR
# As seen in above plots, /not/ a good prediction
print("$" + str(int(lin_reg.predict(6.5))))

# Predict a new result with PR
# More accurate prediction, as seen
print("$" + str(int(lin_reg_2.predict(poly_reg.fit_transform(6.5)))))
