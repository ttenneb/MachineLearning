from distutils.command.build import build
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


def add_split(d_tree, data):
    # calculate the best split
    # calculate means, correlations, and variances of variables
    means = np.mean(data, axis=0)
    corrs = corr(data, means)
    x_split = max(corrs, key=corrs.get)
    # print(corrs, "x split: ", x_split)
    
    # sort and copy vals of selected variable
    vals = np.sort(data[:, x_split])

    # initialize helper variables
    sub_arrays = [[], []]
    min_error = sys.maxsize
    split_val = None
    split_avg = [0, 0]

    # iterate over possible thresholds 
    for val in vals:
        # iterate over the data and split into two sub arrays based on threshold
        for row in data:
            if row[x_split] < val:
                sub_arrays[0].append(row)
            else:
                sub_arrays[1].append(row)
        # calculate the variance from the mean of each sub array
        error = 0
        temp_avg = [0, 0]
        for i, sub_array in enumerate(sub_arrays):
            if len(sub_array) == 0 or len(sub_array) == len(data):
                error = sys.maxsize
                continue
            sub_array =  np.array(sub_array)
            # calculate mean of sub array
            y_mean = np.mean(sub_array[:, -1], axis=0)
            # save the mean incase this is the best split
            temp_avg[i] = y_mean
            # calculate the weighted variance from the mean of the sub array
            for row in sub_array:
                error += ((row[-1] - y_mean)**2)*len(sub_array)/len(data)
        # if this is the best split so far, save the split
        if error < min_error:
            min_error = error
            split_val = val
            split_avg = temp_avg
        sub_arrays = [[], []]
    #  save mean of each sub array, mean of all data, threshold, length of data, and variable to split on
    split_avg.append(means[-1])
    return x_split, split_val, split_avg, len(data)

def split(level, data):
    x_split, thresh_split, _, _ = d_tree[level]
    sub_arrays = [[], []]
    for row in data:
        if row[x_split] < thresh_split:
            sub_arrays[0].append(row)
        else:
            sub_arrays[1].append(row)
    # print("data size: ", len(data), "sub_arrays size: ", len(sub_arrays[0]), len(sub_arrays[1]))
    return sub_arrays

# recursive function to build the decision tree
def build_tree(d_tree, data, depth):
    # check if we have reached max depth (depth is actaully just index of d_tree) log2(depth + 1) == real depth
    if log2(depth + 1) > max_depth:
        return
    # calculate the best split
    x_split, thresh_split, avg, sample_size = add_split(d_tree, data)
    if sample_size == 2:
        print(data)
    print(sample_size, x_split, thresh_split)
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
            
            
            
            if x_split == -1:
                # print("returned at depth ", depth, "max depth ", max_d)
                output.append(avg)
                break
            
    return np.array(output), d_count

d_tree = {}
max_depth = 20
min_sample_size = 1
def main():
    global max_depth
    global d_tree
    global min_sample_size
    

    depth_error = []
    train_depth_error = []

    # generate data
    data = generate(10, 1000)
    test = generate(10, 100)

    # calculate constant and constant prediction calculate error
    mean = np.mean(data, axis=0)[-1]
    print("mean: ", mean)
    c = test[:, -1]
    p = np.ones(c.shape)*mean
    #error between test and sample mean
    print("error: ", np.mean((c-p)**2))

    
    print("Building tree")
    # build a decision tree with train data
    build_tree(d_tree, data, 0)
    print("Tree built")
    d_total = 0
   
    depth_error = []
    train_depth_error = []
    d_total = 0
    for i in range(max_depth):
        max_depth = i
        #test error
        c = (test)[:, -1]
        p, d_count = predict(d_tree, test, max_d=max_depth)
        d_total += d_count
        # error between test and predict
        err = np.mean((c-p)**2)
        depth_error.append((max_depth+1, err))
        print("Finished at depth ", i+1, " with test average error of ", err)
        # train error
        c = (data)[:, -1]
        p, d_count = predict(d_tree, data, max_d=max_depth)
        # error between test and predict
        err = np.mean((c-p)**2)
        train_depth_error.append((max_depth+1, err))
        print("Finished at depth ", i+1, " with train average error of ", err)



    
        
    
    
main()