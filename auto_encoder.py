import numpy as np
from numpy.random import normal
from matplotlib import pyplot as plt
from math import sqrt



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


def tanh(x):
    # max = np.max(x)
    # x = x - max
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def d_tanh(x):
    return 1 - tanh(x)**2


def predict(input, hidden_weights, output_weights):
    x = np.matmul(input, hidden_weights)
    x_tanh = tanh(x)
    x_output = np.matmul(x_tanh, output_weights)
    
    return x_output, [x_tanh, x]


def train(hidden_weights, output_weights, data):
    # Iterate over data
    for i, row in enumerate(data):
        # Predict output and get the hidden states needed for backprop
        output, hidden_state = predict(row, hidden_weights, output_weights)
        # Calculate the gradient of the output weights
        gradient_output = np.matmul(np.reshape((output - row), (-1, 1)), np.reshape(hidden_state[0], (1, -1)))
        # Calculate the gradient of the hidden weights
        gradient_hidden = np.matmul((np.matmul(np.reshape(output-row, (1, -1)), output_weights.T) * d_tanh(hidden_state[1])).T, np.reshape(row, (1, -1)))
        # Update weights 
        output_weights -= gradient_output.T * .001
        hidden_weights -= gradient_hidden.T * .001



# Change in hidden size
# X = generate(5000, .1)
# losses = []
# for i in range(1, 30):
#     hidden_weights = np.random.uniform(size=(30, i), low=-.1, high=.1)
#     output_weights = np.random.uniform(size=(i, 30), low=-.1, high=.1)

#     # for row in X:
#     #     y, _ = predict(row, hidden_weights, output_weights)
#     #     print(y.shape)

#     train(hidden_weights, output_weights, X)
    
#     total_loss = 0
#     for test_row in X:
#         output, _ = predict(test_row, hidden_weights, output_weights)
#         total_loss += np.sum((output - test_row)**2)
#     losses.append(total_loss/5000)
# plt.scatter(range(1, 30), losses)
# plt.show()




# Change in variance
losses = []
for var in [.1, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]:
    X = generate(5000, var)
    
    hidden_weights = np.random.uniform(size=(30, 15), low=-.1, high=.1)
    output_weights = np.random.uniform(size=(15, 30), low=-.1, high=.1)

    # for row in X:
    #     y, _ = predict(row, hidden_weights, output_weights)
    #     print(y.shape)

    train(hidden_weights, output_weights, X)
    
    total_loss = 0
    for test_row in X:
        output, _ = predict(test_row, hidden_weights, output_weights)
        total_loss += np.sum((output - test_row)**2)
    losses.append(total_loss/5000)
plt.scatter([.1, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2], losses)
plt.show()



