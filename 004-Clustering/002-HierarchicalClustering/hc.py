# -*- coding: utf-8 -*-

# Heirarchical Clustering
# Quite similar to K-Means Clustering, often bearing the same results.
#
# Two Types of HC:
# Agglomerative - (Bottom Up Approach)
# Divisive -(Top Up Approach)
#
# Agglomerative HC
# Step 001: Make each data point a single point cluster
# Step 002: Take two closest data points and make them one cluster
# Step 003: Repeat until a single cluster
# (Steps stored in 'memory' of Dendrogram.)
# 
# (Euclidean distance (P1 and P2 = srt(dX**2 +dY**2): distance between 
#  points. Distance between clusters can be taken in four ways: [0] 
#  Closest Points, [1] Further points, [2] Average distance, [3] Distance 
#  between centroids.)

# Import Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import DataSet
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# Plot Dendrogram to Find Optimal Number of Clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# If you look at the included file '/others/dedro_to_optimal_clusters.jpg,
# you'll notice a red line off to the right. This is the largest (euclidean)
# distance without crossing any horizontal lines. This is the area in which
# you place your threshold line (black line). The points at which the threshold
# line crosses (yellow dots) is considered a cluster. Simply count the number
# of yellow dots to obtain your optimal number of clusters.

# Fit HC to Dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

# Visualizing the Clusters
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label='Careful')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label='Standard')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label='Target')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label='Careless')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label='Sensible')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
