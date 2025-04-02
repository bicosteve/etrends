import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression

# Random data samples
# X -> features/independent variables
# y -> target/dependent variables

X = [3, 2, 7, 9, 10, 11, 34, 65, 7]
y = [2, 9, 10, 45, 5, 7, 11, 13, 6]

# Instantiate the linear regression model
model = LinearRegression()


# Train the model
model.fit(X, y)

# Make Predictions

X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

# Plotting the results
plot.scatter(X, y, color="blue", label="Data Points")
plot.plot(X_new, y_predict, color="red", label="Linear Regresion Line")
plot.xlabel("X")
plot.ylabel("y")
plot.title("Linear Regression Test")
plot.legend()
plot.show()
