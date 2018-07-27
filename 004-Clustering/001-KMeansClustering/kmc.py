# -*- coding: utf-8 -*-

# K-Means Clustering
#
# Step 001: Chose number K of clusters
# Step 002: Select random K points, the centroids (not necessarily from DS)
# Step 003: Assign each data point to closest centroid
# Step 004: Compute and place the new centroid for each cluster
# Step 005: Reassign each data point to nearest centroid
# (If any reassignment took place, go to Step 004, else Finished
#
# Think of multiple Pizza Huts in a single city and their delivery radius

# Random Initialization Trap
# The selection of centroids at the beginning of the algorithm
# dictates the final clustering. Incorrect selection leads to incorrect
# clustering. K-Means++ combats this. This will not cover; just, keep
# it in mind.

# Choosing Correct Number of Clusters
# WCSS = a quantifiable metric for ensuring correct cluster num selection.
# This calculates the sum of the centroids distance to each point within
# the cluster squared to find the least value optimal number of clusters.
# At a certain point, the drop in distance is no longer as substantial and
# WCSS marks the optimal number. (See included png.)

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import Dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Using WCSS to Find Optimal Number of Clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
	kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
	kmeans.fit(X)
	wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("WCSS Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
# From the WCSS plot, we notice the optimal number of clusters to be 5

# Apply KMeans to DataSet
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the Clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label='Careful')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label='Standard')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label='Target')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label='Careless')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label='Sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='yellow', label='Centroids')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
