# kmeans to enable us give the lables to the datasets

import sklearn.neighbors as knnx
import sklearn.cluster as kmeansx
from sklearn import metrics
import pandas as pd


data = pd.read_csv("accidents.csv", sep="\t")
x = data[["SensorOne", "Sensor2"]]


kmeansx = kmeansx.KMeans(n_clusters=2)
kmeansx.fit(x)
y = kmeansx.labels_

x_train = x[:6]
x_test = x[6:]
y_train = y[:6]
y_test = y[6:]

knn_one = knnx.KNeighborsClassifier(n_neighbors=1)
knn_three = knnx.KNeighborsClassifier(n_neighbors=3)
knn_five = knnx.KNeighborsClassifier(n_neighbors=5)

knn_one.fit(x_train, y_train)
knn_three.fit(x_train, y_train)
knn_five.fit(x_train, y_train)

y_predict_one = knn_one.predict(x_test)
y_predict_three = knn_one.predict(x_test)
y_predict_five = knn_one.predict(x_test)

accuracy_predict_one = metrics.accuracy_score(y_predict_one, y_test)
accuracy_predict_three = metrics.accuracy_score(y_predict_three, y_test)
accuracy_predict_five = metrics.accuracy_score(y_predict_five, y_test)

# print(y)
# print(kmeansx.cluster_centers_)

print("\n y_actual\t:", y_test)
print("\n y_predict_one\t:", accuracy_predict_one)
print("\n y_predict_three\t:", accuracy_predict_three)
print("\n y_predict_five\t:", accuracy_predict_five)
# print("\n y_actual\t:", y_test)
