from sklearn import tree
from sklearn.tree import export_text
from sklearn import metrics
import numpy

dtree = tree.DecisionTreeClassifier(criterion="entropy")


x_train = numpy.array(
    [
        [6, 5, 7],
        [8, 9, 9],
        [4, 3, 9],
        [7, 3, 8],
        [6, 1, 4],
        [3, 1, 2],
        [9, 10, 9],
        [5, 1, 1],
        [6, 6, 6],
        [5, 7, 4],
    ]
)

y_train = numpy.array(
    [
        "Low Risk",
        "Low Risk",
        "Low Risk",
        "Low Risk",
        "Medium Risk",
        "Medium Risk",
        "Medium Risk",
        "High Risk",
        "High Risk",
        "High Risk",
    ]
)

dtree = dtree.fit(x_train, y_train)

x_test = [[6, 5, 7]]
y_test = numpy.array(["Low Risk"])
y_pred = dtree.predict(x_test)
print(y_pred)

display_tree = export_text(
    dtree, feature_names=["SensorOne", "SensorTwo", "SensorThree"]
)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(display_tree)
print(accuracy)

# most relevant is the root tree here SensorThree
