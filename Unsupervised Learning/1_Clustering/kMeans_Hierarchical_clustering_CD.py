# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:35:44 2022

@author: Chakron.D
"""

#%% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Settings
np.set_printoptions(precision=2)

#%% Importing the dataset

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
print(f'X = {X}')

#%% Optimal number of clusters

# Create subplot
# fig, ax = plt.subplots(1, 3, figsize=(10, 5))
# ax[0].scatter(X[:, 0], X[:, 1], s=10, c=km.labels_)
# ax[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c="r")


# Use the elbow method to find optimal number of clusters [K-Means]
from sklearn.cluster import KMeans
WCSS = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init='k-means++', random_state=42)
    k_means.fit(X)
    WCSS.append(k_means.inertia_) # Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.

# Figure number
plt.figure(0)

# Plot Elbow
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# Dendrogram to find the optimal number of clusters
from scipy.cluster import hierarchy
Z = hierarchy.linkage(X, method='ward') # method=’ward’ uses the Ward variance minimization algorithm

# Plot Dendrogram
plt.subplot(1, 2, 2)
dendrogram = hierarchy.dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Observation Points')
plt.ylabel('Euclidean distance')

#%% Plot clusters

# K-Means cluster
km = KMeans(n_clusters=5, init='k-means++', random_state=0)
km.fit(X)
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# ax[0].scatter(X[:, 0], X[:, 1], s=10, c=km.labels_)
# ax[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=50, c="r")

# plt.figure(1)
# plt.scatter(X[:, 0], X[:, 1], s=50, c=km.labels_)
# plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=100, c="r")

# Figure number
plt.figure(1)

# Plot K-Means cluster
plt.subplot(1, 2, 1)

for i in np.unique(km.labels_):
    plt.scatter(X[km.labels_ == i, 0], X[km.labels_ == i, 1], s=50, label = f'Cluster {i}')
    
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=100, c='k', marker='v', label='Centroid')
plt.legend()
plt.title(f'K-Means Clustering Chart - nCluster = {km.n_clusters}')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')


# Hierarchical cluster
from sklearn.cluster import AgglomerativeClustering

HC = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
HC.fit(X)

# Plot Hierarchical cluster
plt.subplot(1, 2, 2)

for i in np.unique(HC.labels_):
    plt.scatter(X[HC.labels_ == i, 0], X[HC.labels_ == i, 1], s=50, label = f'Cluster {i}')
    
plt.legend()
plt.title(f'Hierarchical Clustering Chart - nCluster = {HC.n_clusters}')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
