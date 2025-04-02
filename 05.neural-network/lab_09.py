# THREE IMAGES

import numpy as np

#  [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1], -> Image One
#  [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1] -> Image Two
# [1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1], -> Image Three


x = [
    [1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1],
    [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1],
    [1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1],
]

[m, n] = np.shape(x)
print(m, n)


y = [[1, 1], [1, -1], [-1, 1]]  # we want to images [], [],[]
print(np.shape(y))
weight = np.zeros((2, n))
weight = np.transpose(weight)

print(weight)


for i in range(m):
    # print(np.shape(x[i]))
    # print(np.shape(y[i]))
    xx = np.reshape(x[i], (n, 1))
    # print(xx)
    yy = np.reshape(y[i], (1, 2))
    # print(yy)
    weight_change_in_w = xx * yy
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

y_in_one = np.dot(test_one, weight)
y_in_two = np.dot(test_two, weight)
print(y_in_one)
print(y_in_two)
# y_in_two = test_two * weight
y_one = np.where(y_in_one >= 0, 1, -1)
y_two = np.where(y_in_two >= 0, 1, -1)
print("Y_in_one ------>", y_in_one)
print("Y_one ------>", y_one)

print("Y_in_two ------>", y_in_two)
print("Y_two ------>", y_two)
