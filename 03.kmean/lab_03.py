import sklearn.cluster as KM
import numpy as np

x_train = [[0, 1], [2, 1], [4, 3], [5, 4]]

kmeansx = KM.KMeans(n_clusters=2)
kmeansx.fit(x_train)
print(kmeansx.labels_)
