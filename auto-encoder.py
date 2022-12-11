import numpy as np
from numpy.random import normal

def generate(n, var):
    x1 = normal(0, 1)
    row = [x1 for i in range(30)]
    for i in range(4, 30, 3):
        row[i-1] += normal(0, var)
    for i in range(2, 30, 3):
        print(i)
        row[i-1] += normal(0, var)
    for i in range(3, 31, 3):
        print(i)
        row[i-1] += normal(0, var)
    return row
        
print(generate(1, .1))