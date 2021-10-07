import streamlit as st
import matplotlib.pyplot as plt
from k_means import kmeans
from generate_data import generate_sample_data, generate_sample_data_centroids
from sklearn.cluster import KMeans


def create_plot_window():
	if 'fig' not in st.session_state:
		st.session_state.fig, st.session_state.ax = plt.subplots(figsize=(14, 12))
		st.session_state.ax.axes.xaxis.set_visible(False)
		st.session_state.ax.axes.yaxis.set_visible(False)
		st.session_state.data_plot = st.pyplot(st.session_state.fig)
	else:
		st.session_state.ax.clear()


# ---------------- Sidebar ---------------------
with st.sidebar.form(key="User input form"):
	st.write("<h1 style='text-align: center'>User Input</h1>", unsafe_allow_html=True)
	st.write("### Choose parameters to draw sample data")

	data_size = st.slider("Data size", min_value=100, max_value=1000)
	data_cluster_size = st.slider('Number of Clusters', min_value=1, max_value=10)
	data_variance = st.slider("Variance", min_value=1, max_value=10)

	generate_button = st.form_submit_button(label="Generate")

with st.sidebar.form(key="Training form"):
	st.write("#### Choose parameter fot training")
	centroid_size = st.slider("Number of Centroids", min_value=1, max_value=10)

	train_button = st.form_submit_button(label="Train")

# --------------- Main page ---------------------
st.write("<h1 style='text-align: center'>K-Means clustering</h1>", unsafe_allow_html=True)
kmeans_description = """
K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.
Typically, unsupervised algorithms make inferences from datasets using only input vectors without referring to known,
or labelled, outcomes. 

You’ll define a target number k, which refers to the number of centroids you need in the dataset.
A centroid is the imaginary or real location representing the center of the cluster.
Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares.

In other words, the K-means algorithm identifies k number of centroids, and then allocates every data point
to the nearest cluster, while keeping the centroids as small as possible.
The ‘means’ in the K-means refers to averaging of the data; that is, finding the centroid.
"""
st.write(kmeans_description)

create_plot_window()

# generate centers for all cluster and plot data around it
if generate_button:
	sample_data_centroids = generate_sample_data_centroids(data_cluster_size, 30, 30)
	st.session_state.sample_data = generate_sample_data(sample_data_centroids, [[data_variance, 0], [0, data_variance]], data_size)
	st.session_state.ax.plot(st.session_state.sample_data[:, 0], st.session_state.sample_data[:, 1], '.', alpha=0.5)
	st.session_state.data_plot.pyplot(st.session_state.fig)

# train the K-Means and color the data
if train_button:
	data = st.session_state.sample_data

	# assignments = kmeans(data, centroid_size)
	assignments = KMeans(n_clusters=centroid_size).fit_predict(data)
	for i in range(centroid_size):
		cluster_i = data[assignments == i]
		st.session_state.ax.scatter(cluster_i[:, 0], cluster_i[:, 1], alpha=0.5)
	st.session_state.data_plot.pyplot(st.session_state.fig)


