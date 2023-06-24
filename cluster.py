import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def dbscan(df, eps, min_samples):
    # Initialize labels array
    labels = np.zeros(len(df))

    # Initialize cluster ID
    cluster_id = 0

    # Iterate through each data point
    for i in range(len(df)):
        # Skip already visited points
        if labels[i] != 0:
            continue

        # Find neighbors within epsilon distance
        neighbors = get_neighbors(df, i, eps)

        # If the number of neighbors is below the threshold, mark as noise (label = -1)
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            cluster_id += 1
            expand_cluster(df, labels, i, neighbors, cluster_id, eps, min_samples)

    return labels


def get_neighbors(df, i, eps):
    # Calculate Euclidean distance between the point i and all other points
    distances = np.linalg.norm(df - df.iloc[i], axis=1)

    # Return indices of points within epsilon distance
    return np.where(distances <= eps)[0]


def expand_cluster(df, labels, i, neighbors, cluster_id, eps, min_samples):
    # Assign cluster ID to the current point
    labels[i] = cluster_id

    # Iterate through each neighbor
    for neighbor in neighbors:
        # Skip already visited neighbors
        if labels[neighbor] != 0:
            continue

        # Find neighbors within epsilon distance of the current neighbor
        new_neighbors = get_neighbors(df, neighbor, eps)

        # If the number of neighbors is above the threshold, add them to the current cluster
        if len(new_neighbors) >= min_samples:
            neighbors = np.concatenate((neighbors, new_neighbors))

        # Assign cluster ID to the current neighbor
        labels[neighbor] = cluster_id

def visualize_clusters(df, labels):
    # Apply PCA to reduce the data to 2 components
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)

    # Retrieve unique cluster labels
    unique_labels = np.unique(labels)

    # Define colors for each cluster
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # Plot each data point with its assigned cluster color
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Plot noise points as black
            color = 'k'
        
        # Select data points belonging to the current cluster
        cluster_points = components[labels == label]
        
        # Plot the cluster points
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=label)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('DBSCAN Clustering with PCA')
    plt.legend()
    plt.show()


