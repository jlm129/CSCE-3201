# Import the required packages
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from ucimlrepo import fetch_ucirepo

# fetch dataset
facebook_live_sellers_in_thailand = fetch_ucirepo(id=488)
# data (as pandas dataframes)
X = facebook_live_sellers_in_thailand.data.features
y = facebook_live_sellers_in_thailand.data.targets
# note that y will be empty as this is a dataset for clustering which does not
# contain labels
# metadata
print(facebook_live_sellers_in_thailand.metadata)
# variable information
print(facebook_live_sellers_in_thailand.variables)
# note that any features that are categorical need to be transformed into
# numeric form as K-Means requires all features to be numeric; Boolean
# features should not be an issue# Define the number of clusters, create the K-Means object (kmeans) using the initialization
# parameters, and train the K-means model with the input data.
num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters,init='k-means++',n_init=10)
kmeans.fit(X)
# Extract and print the centers of the 3 clusters.
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers:")
print(cluster_centers)

# Create two subsets from data and centers from only second and thirdfeatures. 
X_2d = X [:, 1:3] 
cluster_centers_2d = cluster_centers [:, 1:3]
# Step size of the mesh
step_size= 0.01
# Define the grid of points using the sub-dataset X_2d.
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max,step_size),
np.arange(y_min, y_max, step_size))



# Predict the outputs for all the points on the grid using the trained K-means model.

output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
# Plot all output values and color each region.
output = output.reshape(x_vals.shape)
plt.figure() 
plt.clf()
plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(),
y_vals.min(), y_vals.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
# Overlay input data points on top of these colored regions.
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans.labels_, edgecolor='k', s=30)


# Plot the centers of the clusters obtained using the K-Means algorithm.
plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],marker='x', s=200, linewidths=3, color='black', label='Centers')

plt.title("K-Means Clustering Regions and Centers")
plt.legend()
plt.show()
