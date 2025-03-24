import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Load dataset
df = pd.read_csv("kmeans - kmeans_blobs.csv")

# Normalize the dataset (Min-Max Scaling)
def min_max_scaling(data):
    return (data - data.min()) / (data.max() - data.min())

df["x1"] = min_max_scaling(df["x1"])
df["x2"] = min_max_scaling(df["x2"])

data_points = df.to_numpy()

# Function to compute Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# K-Means clustering function
def k_means_clustering(data, k, max_iters=100, tol=1e-4):
    centroids = data[random.sample(range(len(data)), k)]

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        clusters = [np.array(cluster) for cluster in clusters]
        new_centroids = np.array([cluster.mean(axis=0) if len(cluster) > 0 else centroids[i] for i, cluster in enumerate(clusters)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return clusters, centroids

# Run K-Means for k=2 and k=3
clusters_k2, centroids_k2 = k_means_clustering(data_points, k=2)
clusters_k3, centroids_k3 = k_means_clustering(data_points, k=3)

# Function to plot clusters
def plot_clusters(clusters, centroids, k, title):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    plt.figure(figsize=(6, 6))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i}', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=100, label='Centroids')
    plt.xlabel('x1 (Normalized)')
    plt.ylabel('x2 (Normalized)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot clusters for k=2 and k=3
plot_clusters(clusters_k2, centroids_k2, k=2, title="K-Means Clustering (k=2)")
plot_clusters(clusters_k3, centroids_k3, k=3, title="K-Means Clustering (k=3)")
