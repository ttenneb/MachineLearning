import sys
import numpy as np
from numpy.random import normal
from numpy.random import binomial
from math import sqrt, log2, ceil
import matplotlib.pyplot as plt

def generate(d, n):
    x1 = normal(size=(n, 1), loc=3, scale=sqrt(1))
    x2 = normal(size=(n, 1), loc=-2, scale=sqrt(1))
    x3 = x1+2*x2
    x4 = (x2 + 2)**2
    x5 = binomial(1, 0.8, size=(n, 1))
    x = [x1, x2, x3, x4, x5]
    d -= 5
    for i in range(d):
        x.append(normal(size=(n, 1), loc=0, scale=sqrt(1)))
    x = np.concatenate(x, axis=1)
    return calc(x)
def calc(data):
    y = np.zeros((data.shape[0], 1))
    for i, x in enumerate(data):
        y[i] = 4 - 3*x[0]**2 + x[2] - .01*x[3] + x[1]*x[4] + normal(0, .1)
    return np.concatenate((data, y), axis=1)    
   
def var(data, means):
    # calculate the variance of each variable
    totals = {}
    for row in data:
        for i, val in enumerate(row):
            if i not in totals.keys():
                totals[i] = 0
            totals[i] = totals[i] + (val - means[i])**2
    for i in totals.keys():
        totals[i] = totals[i]/(data.shape[0])
    return totals
    
def cov(data, means):
    # calculate the covariance of each variable with y
    totals = {}
    for row in data:
        for i, val in enumerate(row[:-1]):
            if i not in totals.keys():
                totals[i] = 0
            totals[i] = totals[i] + (val - means[i])*(row[-1] - means[-1])
    for i in totals.keys():
        totals[i] = totals[i]/(data.shape[0])
    return totals
def corr(data, means):
    # calculate the correlation of each variable with y using the covariance and variance
    covs = cov(data, means)
    vars = list(var(data, means).values())
    # print("var: ", vars)
    corrs = {}
    # print(len(data), vars)
    for i in range(data.shape[1]-1):
        if i not in corrs.keys():
            corrs[i] = 0
        corrs[i] = abs(covs[i]/sqrt(vars[i]*vars[-1]))
    return corrs



def main():
    # generate data
    d = 10
    data = generate(d, 10000)
    test = generate(d, 1000)

    # learning rate 
    lr = 0.0001
    # initial weights
    w = np.ones((d,))
    # train with SGD
    for i in range(10000):
        # pick random rows
        sample = data[np.random.choice(data.shape[0], 20, replace=False)]
        # calculate the gradient and update weights
        for row in sample:
            w = w - lr*(np.dot(w, row[:-1])-row[-1])*row[:-1]

    # make predictions
    pred = np.dot(test[:, :-1], w)
    # calculate the mean squared error
    mse = np.mean((pred-test[:,-1])**2)
    print("MSE: ", mse)
main()