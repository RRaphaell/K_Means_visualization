import numpy as np
from config import figure


# Compute new center of each cluster
def update_centers(assignments, centers, features, k):
    for i in range(k):
        center_points = features[assignments == i]
        centers[i] = np.mean(center_points, axis=0)


def kmeans(sample_data, centroid_size, data_plot, fig, ax, num_iterations=50):
    """
    This function makes use of numpy functions and broadcasting to speed up the of kmeans algorithm.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2
    """

    N, D = sample_data.shape

    assert N >= centroid_size, 'Number of clusters cannot be greater than number of points'

    # Randomly initialize cluster centers
    center_indexes = np.random.choice(N, size=centroid_size, replace=False)
    centers = sample_data[center_indexes]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iterations):
        scatter = ax.scatter(centers[:, 0], centers[:, 1], s=500)
        for i in range(centroid_size):
            cluster_i = sample_data[assignments == i]
            ax.scatter(cluster_i[:, 0], cluster_i[:, 1], alpha=0.5)
        ax.set(xlim=figure["xlim"], ylim=figure["ylim"])
        data_plot.pyplot(fig)
        scatter.remove()

        features_broadcast = np.tile(sample_data, (centroid_size, 1))
        centers_broadcast = np.repeat(centers, N, axis=0)

        feature_distances = np.linalg.norm(features_broadcast - centers_broadcast, axis=1)
        feature_distances = feature_distances.reshape(centroid_size, N)
        assignments = np.argmin(feature_distances, axis=0)

        center_before_update = centers.copy()
        update_centers(assignments, centers, sample_data, centroid_size)

        # Stop if cluster assignments did not change
        if (center_before_update == centers).all():
            break
