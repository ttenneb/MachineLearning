import sys
import numpy as np
from numpy.random import normal
from numpy.random import binomial
from math import sqrt, log2, ceil
import random
import math
import matplotlib.pyplot as plt

# generate data
def generate(n, var):
    x = [[1 if random.random() < .5 else (-1) for i in range (15)] for i in range(n)]
    x = np.array(x)
    return calc(x, var)


def calc(data, var):
    y = np.zeros((data.shape[0], 1))
    for i, x in enumerate(data):
        y[i] = 0 if 0 < .9*x[0] + (.9**2)*x[1] + (.9**3)*x[2]+(.9**4)*x[3]+(.9**5)*x[4] + normal(0, var) else 1
    return np.concatenate((data, y), axis=1)    
   
# logistic model
def F(w, x):
    y =  1/(1+math.e**(-1*(w[0]+np.dot(w[1:], x[1:])))) 
    if y > random.random():
        return y, 1
    else:
        return y, 0

def main():
    # generate data
    d = 15
    data = generate(10000, .05)
    test = generate(1000, .05)

    # learning rate 
    lr = 0.0001
    # initial weights
    w = np.ones((d+1,))
    # train with SGD
    training_error = []
    testing_error = []
    for i in range(10000):
        # pick random rows
        sample = data[np.random.choice(data.shape[0], 20, replace=False)]
        # calculate the gradient and update weights
        for row in sample:
            # add extra x term to add bias
            x = np.concatenate(([1], row), axis=0)  
            w = w - lr*(F(w, x[:-1])[0]-x[-1])*x[:-1]
        if i % 100 == 0:
            val = []
            # compute predictions
            for row in data[:, :-1]:
                # add extra x term to add bias
                x = np.concatenate(([1], row), axis=0)  
                y = F(w, x)
                val.append(y[0])
            val = np.array(val)
            # store error
            training_error.append((i, np.mean(-1*data[:,-1]*np.log(val) - (1-data[:, -1])*np.log(1-val))))

            val = []
            for row in test[:, :-1]:
                # add extra x term to add bias
                x = np.concatenate(([1], row), axis=0)  
                y = F(w, x)
                val.append(y[0])
            val = np.array(val)
            testing_error.append((i, np.mean(-1*test[:,-1]*np.log(val) - (1-test[:, -1])*np.log(1-val))))
    plt.scatter(*zip(*training_error))
    plt.scatter(*zip(*testing_error))
    plt.show()
    print(testing_error)
    print(training_error)
main()