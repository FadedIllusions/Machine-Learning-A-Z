# -*- coding: utf-8 -*-

# Multiple Linear Regression

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load DataSet
dataset =pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Encode Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])
oneHot = OneHotEncoder(categorical_features=[3])
X = oneHot.fit_transform(X).toarray()

# Avoid 'Dummy Variable Trap'
# Library Does This, Done As Reminder
X = X[:,1:]

# Split DataSet Into Train/Test Set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit MLR to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test Set Results
y_pred = regressor.predict(X_test)

# Optomize Model Via Backward Elimination
# Are Some Independent Variables More Important Than Others?

# Import Needed Library And Add A Column Of Ones To Beginning of X
# Variable So As To Represent The Constant 'b0' In The MLR Equation of
# y = b0 + b1x1 + b2x2 + ... + bnxn
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

# Create Matrix Of Optimal Features That Are Statistically Significant 
# By Removing Non-Statistically Significant Independent Variables
# One-By-One.
#
# Select significance level (to stay in model).
# Fit Model with all possible predictors
# Consider predictor with highest P-Value. If PV > SL, remove predictor
# and fit model without this predictor. (e.g. SL=0.05)
# OLS = Ordinary Least Squares
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# By looking at the summary output, remove the predictor with the highest PV
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Repeat the process...
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
