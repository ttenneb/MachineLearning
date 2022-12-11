from distutils.command.build import build
import sys
import numpy as np
from numpy.random import normal
from numpy.random import binomial
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz 
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

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

def main():
    data = generate(50, 10000)
    test = generate(50, 10)
    clf = tree.DecisionTreeRegressor(max_depth=5)

    clf.fit(data[:,:-1], data[:,-1])
    p = clf.predict(test[:,:-1])
    c = (test)[:, -1]
    print(c, p)
    err = np.mean((c-p)**2)
    print("Error: ", err)

    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data) 
    graph.render("tree") 
    # mean = np.mean(data, axis=0)[-1]
    # print("mean: ", mean)
    # p_min = sys.maxsize
    # p_min_i = 0
    # c = test[:, -1]
    # for i in range(int(abs(mean*2))*-1, int(abs(mean*2))):
    #     p = np.ones(c.shape)*i
    #     err = np.mean((c-p)**2)
    #     if err < p_min:
    #         p_min = err
    #         p_min_i = i
    # print("p_min_i: ", p_min_i, "error: ", np.mean((c-p_min_i)**2))


main()