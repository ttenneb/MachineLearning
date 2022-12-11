from distutils.command.build import build
import sys
import numpy as np
import math
import random
from numpy.random import normal
from numpy.random import binomial
from math import sqrt, log2, ceil
import matplotlib.pyplot as plt

def generate(n, var):
    x = [[1 if random.random() < .5 else (-1) for i in range (15)] for i in range(n)]
    x = np.array(x)
    return calc(x, var)


def calc(data, var):
    y = np.zeros((data.shape[0], 1))
    for i, x in enumerate(data):
        y[i] = -1 if 0 < .9*x[0] + (.9**2)*x[1] + (.9**3)*x[2]+(.9**4)*x[3]+(.9**5)*x[4] + normal(0, var) else 1
    return np.concatenate((data, y), axis=1)    
    
# information gain of all X_i on Y in data
def information_gain(data):
    # P(Y = y)
    if len(data) == 0:
        return
    Py = {}
    total_neg = 0
    total_pos = 0
    for row in data:
        if row[-1] == -1.0:
            total_neg  += 1
        else:
            total_pos +=1
    Py[-1] = total_neg/len(data)
    Py[1] = total_pos/len(data)
    
    if Py[1] == 0 or Py[1] == 1:
        return

    total_x = {}
    # Initialize dict
    for x in [-1, 1]:
        for i in range(len(data[0][:-1])):
            total_x[(i, x)] = 0

    total_xy = {}
    # Initialize dict
    for x in [-1, 1]:
        for y in [-1, 1]:
            for i in range(len(data[0][:-1])):
                total_xy[(i, x, y)] = 0
    # count totals
    for row in data:
        y = row[-1]
        for i, x in enumerate(row[:-1]):
            total_x[(i, x)] += 1 
            total_xy[(i,x,y)] += 1
    # estimate probabilities
    # P(X_i=x)
    Px = {}
    for x in [-1, 1]:
        for i in range(len(data[0][:-1])):
            if (total_x[(i, -1)] + total_x[(i, 1)]) != 0:
                Px[(i, x)] = total_x[(i, x)]/(total_x[(i, -1)] + total_x[(i, 1)])
            else:
                Px[(i, x)] = 0

    # P(Y = y | X_i = x)
    Pxy = {}
    # Initialize dict
    for x in [-1, 1]:
        for y in [-1, 1]:
            for i in range(len(data[0][:-1])):
                if total_x[(i, x)] != 0:
                    Pxy[(i, x, y)] = total_xy[(i, x, y)]/total_x[(i, x)]
                else:
                    Pxy[(i, x, y)] = 0
    Hy = 0
    for y in [-1, 1]:
        Hy += Py[y]*math.log2(Py[y]+0.00000001)
    Hy = -1*Hy

    IG = {}
    for i in range(len(data[0][:-1])):
        # H(Y|X_i)
        totalx = 0
        for x in [-1, 1]:
            totaly = 0 
            for y in [-1, 1]:
                totaly += Pxy[(i, x, y)]*math.log2(Pxy[(i, x, y)]+0.00000001)
            totalx += -1*totaly*Px[(i, x)]
        IG[i] = Hy - totalx
    # Where IG[i] is the information gain of X_{i-1}
    return IG

def add_split(d_tree, data):
    # calc information gain
    ig = information_gain(data)
    if ig == None:
        return -1,0,0,0
    # find the X_i with most info on y
    x_split = max(ig, key=ig.get)
    # print("adding split, data size: ", len(data))
    split_val = 0

    # Calc mean of all this data (use the mean to find the majorty vote)
    means = [0,0,0]
    means[2] = np.mean(data[:-1])
    if means[2] > 0:
        means[2] = 1
    else:
        means[2] = -1

    # split data into two subarrays and find their means incase it's a terminating node
    sub_arrays = [[], []]
    for row in data:
            if row[x_split] < split_val:
                sub_arrays[0].append(row)
            else:
                sub_arrays[1].append(row)

    for i, arr in enumerate(sub_arrays):
        if len(arr) < 1:
            means[i] = .5
            continue
        arr = np.array(arr)
        means[i] = np.mean(arr[:, -1], axis=0)
        if means[i] > 0:
            means[i] = 1
        else:
            means[i] = -1
    # return the x_i split on, the split value (0), the means, and the sample size at this leaf
    return x_split, split_val, means, len(data)

# splits the data at level d of the decision tree into two groups
def split(level, data):
    x_split, thresh_split, _, _ = d_tree[level]
    sub_arrays = [[], []]
    for row in data:
        if row[x_split] < thresh_split:
            sub_arrays[0].append(row)
        else:
            sub_arrays[1].append(row)
    return sub_arrays

# recursive function to build the decision tree
def build_tree(d_tree, data, depth):
    # check if we have reached max depth (depth is actaully just index of d_tree) log2(depth + 1) == real depth
    if log2(depth + 1) > max_depth:
        return
    # calculate the best split
    x_split, thresh_split, avg, sample_size = add_split(d_tree, data)
    if x_split == -1:
        return

    if sample_size < 2:
        return
    # create node in tree
    if depth not in d_tree.keys():
        d_tree[depth] = [-1, -1, None, -1]
    # add data to node
    d_tree[depth] = (x_split, thresh_split, avg[2] if d_tree[depth][2] is None else d_tree[depth][2], sample_size)
    
    # create children nodes if its not too deep
    if(log2(depth*2 + 1 + 1) < max_depth):
        d_tree[depth*2 + 1] = (-1, -1, avg[0], -1)
    if(log2(depth*2 + 2 + 1) < max_depth):
        d_tree[depth*2 + 2] = (-1, -1, avg[1], -1)
    # print("split at, ", depth)
    data1, data2 = split(depth, data)
    data1 = np.array(data1)
    data2 = np.array(data2)
    # continue building tree on children nodes
    # print("node: ", depth)
    if len(data1) > 1:
        build_tree(d_tree, data1, depth*2 + 1)
    if len(data2) > 1:
        build_tree(d_tree, data2, depth*2 + 2)


def predict(d_tree, test, max_d):
    global min_sample_size
    output = []
    # predict the output for each row in the test data
    # count the number of irrelavent features used on this data
    d_count = 0
    for data in test:
        depth = 0
        while True:
            # continue down tree until terminating condition is met
            x_split, thresh_split, avg, sample_size = d_tree[depth]
            if x_split > 5:
                d_count += 1
            if data[x_split] < thresh_split:
                d = depth*2 + 1
            else:
                d = depth*2 + 2
            
            #  terminating condition 1: min sample size
            if sample_size == -1 or sample_size < min_sample_size:
                output.append(avg)
                break

            #  terminating condition 2: max depth
            if ceil(log2(d+1)) > max_d:
                # print("returned at depth ", depth, "max depth ", max_d)
                output.append(avg)
                break
            else:
                depth = d
            
            # leaf node
            if x_split == -1:
                output.append(avg)
                break
            
    return np.array(output), d_count

d_tree = {}
max_depth = 50
min_sample_size = 2500
def main():
    global max_depth
    global d_tree
    global min_sample_size
    

    depth_error = []
    train_depth_error = []
    # generate data
    data = generate(5000, 2)
    test = generate(500, 2)

    # calculate constant and constant prediction calculate error (we want to know if this model even does anything)
    mean = np.mean(data, axis=0)[-1]
    if mean > 0:
        mean = 1
    else:
        mean = -1
    print("mean: ", mean)
    c = test[:, -1]
    p = np.ones(c.shape)*mean
    #error between test and sample mean
    total_miss = 0
    for j,x in enumerate(p):
        if x != c[j]:
            total_miss += 1
    print("total_miss: ", total_miss)

    
    print("Building tree")
    # build a decision tree with train data
    build_tree(d_tree, data, 0)
    print("Tree built")
    d_total = 0
   
    depth_error = []
    train_depth_error = []
    d_total = 0
    for i in range(1, min_sample_size, 100):
        min_sample_size = i
        #test error
        c = (test)[:, -1]
        p, d_count = predict(d_tree, test, max_d=max_depth)
        d_total += d_count
        # error between test and predict
        total_miss = 0
        for j,x in enumerate(p):
            if x != c[j]:
                total_miss += 1
        depth_error.append((min_sample_size, total_miss/500)) 
        print("Finished at min_sample_size ", i, " with total_miss of ON TEST", total_miss/500)
        # train error
        c = (data)[:, -1]
        p, d_count = predict(d_tree, data, max_d=max_depth)
        # error between train and predict
        total_miss = 0
        for j,x in enumerate(p):
            if x != c[j]:
                total_miss += 1
        train_depth_error.append((min_sample_size, total_miss/5000))
        print("Finished at min_sample_size ", i, " with total_miss of ON TRAIN", total_miss/5000)
    plt.scatter(*zip(*depth_error))
    plt.scatter(*zip(*train_depth_error))
    plt.show()

main()