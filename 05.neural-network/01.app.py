import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv("instanbul_stock_exchange.csv", sep=",")

X = data[["SP"]]
y = X.shift(5)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


model = keras.Sequential([keras.layers.Dense(1, activation="linear", input_shape=(1,))])
# Single neuron (Perception)

model.add(keras.Input(shape=(2,)))
model.add(keras.layers.Dense(8))

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual ISE Values")
plt.xlabel("Predicted ISE Values")
plt.title("Actual vs Predicted ISE index valuse")
plt.show()
