import numpy as np
from numpy.random import normal
from numpy.linalg import eig
from matplotlib import pyplot as plt

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

# Change with variance
fig, axs = plt.subplots(3, 3)
for i, var in enumerate([.1, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]):
    X = generate(5000, var)
    covar = np.matmul(X.T, X)
    eigenvalues, eigenvectors = eig(covar)

    eigenpairs = list(zip(eigenvalues, eigenvectors))
    eigenpairs.sort(key=lambda x: x[0], reverse=True)

    pc1 = eigenpairs[0][1] # first principal component
    pc2 = eigenpairs[1][1] # second principal component

    # data along the two principal components

    data1 = np.matmul(X, pc1)
    data2 = np.matmul(X, pc2)

    # Plot the data

    axs[i//3, i % 3].scatter(data1, data2)
    axs[i//3, i % 3].set_title("Variance: " + str(var))
plt.show()

