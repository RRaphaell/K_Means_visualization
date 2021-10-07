import numpy as np


def kmeans(features, k, num_iterations=100):
    """
    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    center_indexes = np.random.choice(N, size=k, replace=False)
    centers = features[center_indexes]
    assignments = np.zeros(N, dtype=np.uint32)

    # Compute new center of each cluster
    def update_centers(assignments, centers, features, k):
        for i in range(k):
            center_points = features[assignments == i]
            centers[i] = np.mean(center_points, axis=0)

    for n in range(num_iterations):
        features_broadcast = np.tile(features, (k, 1))
        centers_broadcast = np.repeat(centers, N, axis=0)

        feature_distances = np.linalg.norm(features_broadcast - centers_broadcast, axis=1)
        feature_distances = feature_distances.reshape(k, N)
        assignments = np.argmin(feature_distances, axis=0)

        center_before_update = centers.copy()
        update_centers(assignments, centers, features, k)

        # Stop if cluster assignments did not change
        if (center_before_update == centers).all():
            break

    return assignments
