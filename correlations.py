import numpy as np 
from numpy.random import normal
import networkx as nx
import matplotlib.pyplot as plt

def fit_model(data, predict_index, lr):
    w = np.ones((data.shape[1]-1,))
    # select y from the data
    y = data[:, predict_index]
    # remove y from the datac
    data = np.delete(data, predict_index, axis=1)
    # add y to the end of the data
    data = np.append(data, np.reshape(y, (-1, 1)), axis=1)
    for i in range(10000):
        # pick random rows
        sample = data[np.random.choice(data.shape[0], 20, replace=False)]
        # calculate the gradient and update weights
        for row in sample:
            w = w - lr*(np.dot(w, row[:-1])-row[-1])*row[:-1]
    pred = np.dot(data[:, :-1], w)
    # calculate the mean squared error
    mse = np.mean((pred-data[:,-1])**2)
    
    return w, mse

def generate(n, var):
    return np.array([generate_row(var) for i in range(n)])
def generate_row(var):
    x1 = normal(0, 1)
    row = [x1 for i in range(30)]
    for i in range(4, 30, 3):
        row[i-1] += normal(0, var)
    for i in range(2, 30, 3):
        row[i-1] += normal(0, var)
    for i in range(3, 31, 3):
        row[i-1] += normal(0, var)
    return np.array(row) 

# Correlation graph with number of top features selected or threshold 

# Change the number of features to select for in the grap 
X = generate(5000, 0.1)

# define graph for visualization
G = nx.Graph()

# add a node for each var [1, 31)
G.add_nodes_from(range(1, 31))

# number of features to include in the graph
graph_features = 4
threshold = 0.033

count_true_dependency = 0
for i in range(30):
    w, mse = fit_model(X, i, .0001)

    variables = list(range(1,31))
    # remove the variable we are predicting
    variables.remove(i+1)
    # label each weight with its repsective variable
    w = zip(w, variables)
    # sort w 
    w = sorted(w, key=lambda x: x[0], reverse=True)

    # add top edges to graph based number of graph features or threshold
    if graph_features == -1: 
        for j in w[:sum(i[0] > threshold for i in w)]:
            G.add_edge(i+1, j[1])
            if j[1] % 3 == i % 3:
                count_true_dependency += 1
    else:
        for j in w[:graph_features]:
            G.add_edge(i+1, j[1])
            if j[1] % 3 == i % 3:
                count_true_dependency += 1

# Draw the graph
options = {
    'arrowstyle': '-|>',
    'arrowsize': 12,
}
print(count_true_dependency/G.number_of_edges())
nx.draw_networkx(G, arrows=True, **options)
plt.show()






# Correlation graph as variance changes

# Change the number of features to select for in the grap 
# depedency_percent = []
# for var in [.1, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]:
#     X = generate(1000, var)

#     G = nx.Graph()

#     G.add_nodes_from(range(1, 31))

#     # number of features to include in the graph
#     graph_features = -1
#     threshold = 0.04

#     count_true_dependency = 0
#     for i in range(30):
#         w, mse = fit_model(X, i, .0001)

#         variables = list(range(1,31))
#         # remove the variable we are predicting
#         variables.remove(i+1)
#         print(variables)
#         # label each weight with its repsective variable
#         w = zip(w, variables)
#         # sort w 
#         w = sorted(w, key=lambda x: x[0], reverse=True)

#         # add top edges to graph based number of graph features or threshold
#         if graph_features == -1: 
#             for j in w[:sum(i[0] > threshold for i in w)]:
#                 G.add_edge(i+1, j[1])
#                 if j[1] % 3 == i % 3:
#                     count_true_dependency += 1
#         else:
#             for j in w[:graph_features]:
#                 G.add_edge(i+1, j[1])
#                 if j[1] % 3 == i % 3:
#                     count_true_dependency += 1
#     depedency_percent += [count_true_dependency]
    

# # Plot number of true depedencies as a function of variance
# plt.scatter([.1, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2], depedency_percent)
# plt.show()