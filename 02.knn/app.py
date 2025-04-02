import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


"""
1. Initializing the header files
numpy is for creating array of the dataset
sklearn.model_selection train_test_split for splitting the data
sklearn.neighbours KNeighborsClassifier creating the model instance
sklearn.metrics accuracy_score, classification_report for getting the accuracy and classification report
"""

# 2. X dataset
X = np.array(
    [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 3.0],
        [2.5, 4.5],
        [3.5, 5.0],
        [6.0, 5.0],
        [7.0, 7.0],
        [5.5, 6.5],
        [7.0, 4.5],
        [8.0, 6.0],
    ]
)

# 3. Y dataset
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# 4. Splitting the data set into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40
)

knn_model = KNeighborsClassifier(3)

# 4. Train the model
knn_model.fit(X_train, y_train)

# 5. Make predictions on test set using X_test
predictions = knn_model.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"Classification report {report}")

# 7. Prediction for new datapoints
new_data = np.array([[4, 4], [7.5, 5.5]])
prediction = knn_model.predict(new_data)
print(f"Prediction for new data is {prediction}")
