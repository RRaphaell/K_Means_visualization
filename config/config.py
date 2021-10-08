sidebar = dict(
    title='<h1 style="text-align: center">User Input</h1>',
    data_title='### Choose parameters to draw sample data',
    train_title='### Choose parameters for training',
    data=dict(
        size=(100, 1000),
        cluster_size=(1, 10),
        variance=(1, 10),
    ),
    train=dict(
        centroid_size=(1, 10),
    )
)

main_page = dict(
    title="<h1 style='text-align: center'>K-Means clustering</h1>",
    description="""K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.
    Typically, unsupervised algorithms make inferences from datasets using only input vectors without referring to 
    known, or labelled, outcomes. 
    \nYou’ll define a target number k, which refers to the number of centroids you need in the dataset.
    A centroid is the imaginary or real location representing the center of the cluster.
    Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares.
    \nIn other words, the K-means algorithm identifies k number of centroids, and then allocates every data point
    to the nearest cluster, while keeping the centroids as small as possible.""",
    data_size=(100, 1000),
    data_cluster_size=(1, 10),
    data_variance=(1, 10),
)

figure = dict(
    figsize=(14, 12),
    xlim=(0, 30),
    ylim=(0, 30),
    offset=2,
)
