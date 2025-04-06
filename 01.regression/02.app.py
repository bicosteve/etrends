import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


data = pd.read_csv("instanbul_stock_exchange.csv", sep=",")

X = data[["SP"]]
y = X.shift(5)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


model = LinearRegression()

model.fit(X_train, y_train)

print(f"Coefficient {model.coef_}")
print(f"Intercept {model.intercept_}")

predictions = model.predict(X_test)

# Evalueate the model

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error {mse}")
print(f"R2 {r2}")
