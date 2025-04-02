# 1. Imports and header files
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 2. X which is a 2d array of the values to use
X = np.array([2, 4, 12, 3, 30, 20, 11, 25, 1, 15, 18]).reshape(-1, 1)
# Reshape is done because sklearn expects the data to be in 2D array

# 3. the number of clusters
k = 3

# 4. Initialize the Kmeans model

kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
# -> Line 16 is to create the model and n_init is used to avoid warnings.


# 5. Train the model
# Standardize the data with scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_model.fit(X_scaled)

# 6. Get the cluster labels for each data point
labels = kmeans_model.labels_


# 7. Get the cluster centers
centers = kmeans_model.cluster_centers_


# 8. Print the cluster labels and centers

print(f"Cluster labels {labels}")
print(f"Cluster centers {centers}")

# 9. Visualize the results
# (i) Plots the data points
plt.scatter(X_scaled, [0] * len(X), c=labels, cmap="viridis")
# (ii) Plot the centroids
plt.scatter(centers, [0] * len(centers), marker="X", s=200, c="red", label="Centroids")
# (iii) Remove the y ticks since it is 1D data
plt.yticks([])
# (iv) Xlabels
plt.xlabel("Data Values")
# (v) Title
plt.title(f"K-Means Clustering (k={k})")
plt.legend()
plt.show()

# 10. Making predictions with new data
new_data = np.array([[16], [8], [32]])
new_labels = kmeans_model.predict(new_data)
print(f"New Data Predictions {new_labels}")


# 11. Accessing individual clusters data
for i in range(k):
    cluster_points = X_scaled[labels == i]
    print(f"Cluster {i} points {cluster_points.flatten()}")
