# 2 IMAGE

import numpy as np

#  [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1], -> Image One
#  [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1] -> Image Two


x = [
    [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
    [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
]

[m, n] = np.shape(x)
print(m, n)


y = [1, -1]  # we want to images [], []
# y = [1, -1, -1, -1]

weight = [0] * n
print(weight)

n = len(y)

for i in range(m):
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

test_one = [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1]  # distort
test_two = [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1]  # distort
y_in_one = np.sum(test_one * weight)
y_in_two = np.sum(test_two * weight)
y_one = np.where(y_in_one >= 0, 1, -1)
y_two = np.where(y_in_two >= 0, 1, -1)
print("Y_in_one ------>", y_in_one)
print("Y_one ------>", y_one)

print("Y_in_two ------>", y_in_two)
print("Y_two ------>", y_two)
