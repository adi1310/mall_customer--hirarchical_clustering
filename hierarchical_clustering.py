# Hierarchical Clustering

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset using pandas
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrograms')
plt.xlabel('Clusters')
plt.ylabel('Eiclidean Distances')
plt.show()

# Fitting hierarchial clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

# Visualising the Clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 50, c = 'red',     label = 'careful')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 50, c = 'blue',    label = 'standard')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 50, c = 'green',   label = 'target')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 50, c = 'cyan',    label = 'careless')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 50, c = 'magenta', label = 'sensable')
plt.title('clustet of clients')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show() 
