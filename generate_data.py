import numpy as np
from collections import namedtuple
from typing import NamedTuple, List


def generate_sample_data(means: List[NamedTuple], cov: List[List], n_data: int = 200) -> np.ndarray:
	data = np.empty([n_data, 2])
	for mean in means:
		arr = np.random.multivariate_normal([mean.x, mean.y], cov, n_data)
		data = np.vstack([data, arr])

	return data
	
	
def generate_sample_data_centroids(number_of_centroids: int, row: int = 10, col: int = 10) -> List[NamedTuple]:
	centroids = []
	available_centroids = np.ones((row, col))

	centroid = namedtuple('centroid', 'x y')
	for i in range(number_of_centroids):
		centroid_x, centroid_y = np.random.randint(0, row), np.random.randint(0, col)
		if available_centroids[centroid_x, centroid_y]:
			new_centroid = centroid(centroid_x, centroid_y)
			centroids.append(new_centroid)
			available_centroids[centroid_x, centroid_y] = False
	return centroids
