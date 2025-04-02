# AND

import numpy as np

x = [[1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]]
y = [1, -1, -1, -1]

weight = [0] * 3
print(weight)

n = len(y)

for i in range(n):
    weight_change_in_w = np.transpose(x[i]) * y[i]
    weight = weight + weight_change_in_w
    print(
        "x ----->",
        x[i],
        "y ----->",
        y,
        "small_change_in_weight----->",
        weight_change_in_w,
    )

test = [1, -1, 1]
y_in = np.sum(test * weight)
y = np.where(y_in >= 0, 1, -1)
print("Y_in ------>", y_in)
print("Y ------>", y)
