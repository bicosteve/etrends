t = int(input("Enter time:\t "))
x = 0
p = 100
a = p

while x < t:
    a = 1.1 * a
    print("Year: \t", t, "\t Amount:\t", a)
    x = x + 1

# Arrays
import numpy
from matplotlib import pyplot  # for plotting points
from scipy import stats as sc  # regression

x = [4, 7, 3, 6, 10, 5]
y = [2, 5, 1, 5, 2, 4]

g, c, r, p, std = sc.linregress(x, y)

y2 = [g * xi + c for xi in x]

print(x, "\t", y, "\n", y2)

pyplot.scatter(x, y, color="blue", marker="o")
pyplot.plot(x, y, color="blue", marker="o")
pyplot.xlabel("X Values")
pyplot.ylabel("Y Values")
pyplot.show()
