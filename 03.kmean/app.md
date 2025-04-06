What is KMean Clustering
-> This is unsupervised machne learning algorithm for clustering data.
-> K-means aims to partition the dataset into 'k' distinct, non-overlapping subsets (clusters)
where each datapoint belongs the cluster with the nearest mean (cluster center or centroid)
-> K-means will attempt to group similar clusters of data together
-> A typical clustering problem looks like

- cluster similar documents
- cluster customers based on features
- market segmentation
- identify similar groups

-> overall goal is to divide data into distinct groups so that the observation are similar.
How does clutering work

- Choose a number of clusters 'k'
- Randomly assign each point to a cluster
- Untill clusters stop changing, repeat the following
  for each cluster compute the centroid by taking the mean vector points in the cluster.
  assign each data point to the cluster for which the centroid is the closest.
  -> choosing a k value is not simple.
  -> there are various ways to choose the k value and one of them is elbow method.
  -> you can first compute the sum of the squared error SSE for some values of k eg 2,4,6,8
  -> SSE is defined as the sum of the squared distance between each member of the cluster and its centroid.
  ->
