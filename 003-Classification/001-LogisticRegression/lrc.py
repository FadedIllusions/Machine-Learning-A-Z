#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Logistic Regression Classification

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import DataSet
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

# Train/Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fit Logistic Regression to Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict Results
y_pred = classifier.predict(X_test)

# Make Confusion Matrix -- For Evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Type in 'cm' in console to print confusion matrix
# ([65, 3]  65+24 = 89 (Correct Predictions)
#  [8, 24]) 8+3 = 11 (Incorrect Predictions)

# Visualizing the Training Set Results -- For Evaluation
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

## ---  --  - Interpretting the Visualization -  --  --- ##
#
# Red Dots (Didn't Buy), Green Dots (Did Buy)
# Colored Sections of Graph = 'Prediction Regions' (Most Likely)
# Thus, the dots represent the 'true' buyers and regions represent 
# classified/predicted regions of those that won't buy and will buy.
#
# This classifier is Linear; So, the prediction boundary seperater
# a straight line.

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

## ---  --  - How The Visualization Code Works -  --  --- ##
#
# Essentially, we use the prediction boundary seperater of the
# classification model to split the screen into either 0 or 1
# (won't buy, will buy, based on the classification prediction).
# We, then, apply the contour so as to actually color those areas
# red and green, respectively. Afterwhich we iterate through the
# data and plot each point per whether they actually bought or not,
# coloring them respectively.
