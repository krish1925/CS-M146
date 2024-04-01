# Part (a)
import numpy as np
from numpy.linalg import norm
import statistics as st
import matplotlib.pyplot as plt

pts = np.array([
    [0, 5],
    [1, 4],
    [3, 7],
    [-2, 1],
    [-1, 13],
    [10, 3],
    [12, 7],
    [-7, -1],
    [-3, 12],
    [5, 9],
])

val = ([0, 0, 1, 1, 0, 0, 1, 0, 1, 1])

def func(x_1, j_1):
    if x_1 - j_1 < 0:
        return 0
    if x_1 - j_1 > 0:
        return 1

low = -8
high = 13
err = []
j_1 = []
for i in range(low, high + 1):
    err_rate = 0
    for j in range(0, 10):
        if func(pts[j][0], i) != val[j]:
            err_rate += 0.1
    err.append(err_rate)
    j_1.append(i)
    print(err_rate, i)

print(err)
print(j_1)

# Part (b)
low = -2
high = 14
err = []
j_2 = []
for i in range(low, high + 1):
    err_rate = 0
    for j in range(0, 10):
        if func(pts[j][1], i) != val[j]:
            err_rate += 0.1
    err.append(err_rate)
    j_2.append(i)
    print(err_rate, i)

print(err)
print(j_2)


# Part (d)
weights = [
    0.0625, 0.0625, 0.0625, 0.25, 0.25,
    0.0625, 0.0625, 0.0625, 0.0625, 0.0625
]
low = -8
high = 13
err = []
j_1 = []
for i in range(low, high + 1):
    err_rate = 0
    for j in range(0, 10):
        if func(pts[j][0], i) != val[j]:
            err_rate += weights[j]
    err.append(err_rate)
    j_1.append(i)
    print(err_rate, i)

print(err)
print(j_1)

