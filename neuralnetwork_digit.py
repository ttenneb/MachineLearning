#%%
import numpy as np
import math
import random
from numpy.core.numeric import outer
from numpy.random.mtrand import rand
import matplotlib.pyplot as plt
import timeit

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, numHidden, learning_rate) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.numHidden = numHidden
        
        self.lr = learning_rate
        
        self.weights = [None]*(numHidden+1)
        self.bias = [None]*(numHidden+1)
        
        
        for i, weight in enumerate(self.weights):
            self.weights[i] = (np.random.rand(hidden_size, hidden_size)*2)-1
        self.weights[0] = (np.random.rand(hidden_size, input_size)*2)-1
        self.weights[-1] = (np.random.rand(output_size, hidden_size)*2)-1
        
        for i, b in enumerate(self.bias):
            self.bias[i] = np.random.rand(hidden_size, 1)
        self.bias[-1] = np.random.rand(output_size, 1)
    def sigmoid(self, input):
        output = np.ndarray(np.shape(input))
        output.fill(math.exp(1))  
        
        output = np.power(output, input*-1)
        output = output + 1
        
        temp = np.ndarray(np.shape(input))
        temp.fill(1)
        
        output = temp/output
        return output
    def d_sigmoid(self, input):
        # output = np.ndarray(np.shape(input))
        # output = (input*-1)+1
        # print(output)
        # output = output*input
        
        # print(output)
        # return output
        return self.sigmoid(input)*(1-self.sigmoid(input))
    def predict(self, input):
        hidden = [None]*self.numHidden
        output = np.ndarray((self.output_size, 1))
        sum = np.matmul(self.weights[0], input) + self.bias[0]
        hidden[0] = self.sigmoid(sum)
        for i, h in enumerate(hidden[1:], 1):
            sum = np.matmul(self.weights[i], hidden[i-1]) + self.bias[i]
            hidden[i] = self.sigmoid(sum)
        output=self.sigmoid(np.matmul(self.weights[self.numHidden], hidden[-1]) + self.bias[self.numHidden])
        return (output, hidden)
    
    
    def train(self, input, target, test, test_labels):
        accuracy = []
        time = []
        for sample in range(10):
            start = timeit.default_timer()
            for i, d in enumerate(input[:int((len(input)/10)*sample)]):
                output, hidden = self.predict(d)
                
                error = [None]*(self.numHidden+1)
                gradient = [None]*(self.numHidden+1)
                deltaM = [None]*(self.numHidden+1)
                
                error[0] = target[i] - output
                for j, e in enumerate(error[1:], 1):
                    error[j]=np.matmul(self.weights[-j].transpose(), (error[j-1]))

                gradient[0]=error[0] * (self.d_sigmoid(output)) * self.lr
                deltaM[0] = np.matmul(gradient[0], (hidden[self.numHidden-1].transpose()))
                for j, g in enumerate(gradient[1:], 1):
                    gradient[j] =error[j] * (self.d_sigmoid(hidden[self.numHidden-j-1])) * self.lr
                    deltaM[j]= np.matmul(gradient[j], hidden[self.numHidden-j-1].transpose())

                gradient[self.numHidden]=error[self.numHidden] * (self.d_sigmoid(hidden[0])) * self.lr
                deltaM[self.numHidden]=np.matmul(gradient[self.numHidden], (d.transpose()))
                
                for j, g in enumerate(gradient):
                    self.bias[j] += gradient[self.numHidden - j]
                    self.weights[j] += deltaM[self.numHidden - j]
            stop = timeit.default_timer()
            count = 0
            for i, d in enumerate(test):
                output, _ = self.predict(d)
                guess = np.argmax(output)
                if guess == int(test_labels[i]):
                    count += 1    
            a = count / len(test_labels)
            print(a)
            time.append(stop - start)
            accuracy.append(a)
        return accuracy, time
def convert_data(str):  # this function converts the images into a 21 x 28 array
    np.set_printoptions(linewidth=np.inf)
    with open(str) as f:  # reads in line by line
        lines = f.readlines()

    data = []
    count = 0
    count_rows = 0
    data.append(np.zeros(shape=(28, 28)))
    for line in lines:
        if not line.isspace():
            for i, c in enumerate(line[:-1]):
                if c == '+':
                    data[-1].itemset((count, i), 1)
                if c == '#':
                    data[-1].itemset((count, i), 2)
        count += 1
        count_rows += 1
        if (count_rows == 28):
            count = 0
            count_rows = 0
            data.append(np.zeros(shape=(28, 28)))
    return data[:-1]
                
                
def pixel_features(input, size):
        features = [None]*(size*size)
        for i, row in enumerate(input):
            for j, x in enumerate(row):
                features[j + i*size] = x
        return np.array(np.array(features)).reshape(784,1)
    
    
def convert_label(
        str):  # this function creates an array of the matching labels to go with the data index 0 --> 0, 1 --> 1, etc...
    with open(str) as f:  # reads in line by line
        lines = f.readlines()
        labels = [None] * len(lines)

        i = 0
        for line in lines:
            line = line.strip("\n")
            labels[i] = line
            i += 1
        return labels    
      
      
model = NeuralNetwork(784, 128, 10, 1, .1)

training_data = convert_data('data/digitdata/trainingimages')  # gets training data into numpy array format
training_labels = convert_label('data/digitdata/traininglabels')  # gets training label into array format
validation_data = convert_data('data/digitdata/validationimages')  # gets training data into numpy array format
validation_labels = convert_label('data/digitdata/validationlabels')  #
test_data = np.array(convert_data('data/digitdata/testimages'))
test_labels = convert_label('data/digitdata/testlabels')

training_data = np.array(training_data)

train_input = []
train_labels_input = []
test_input = []
test_labels_input = []

train_amount = 20000
for i in range(train_amount):
    select = random.randint(0, len(training_data)-1)
    train_input.append(pixel_features(training_data[select], 28))
    
    one_hot = np.zeros((10, 1))
    one_hot[int(training_labels[select]), 0] = 1
    train_labels_input.append(one_hot)
   

for i in test_data:
    test_input.append(pixel_features(i, 28))

accuracy = []
time = []
for i in range(5):
    model = NeuralNetwork(784, 128, 10, 1, .1)
    a, t = model.train(train_input, train_labels_input, test_input, test_labels)
    accuracy.append(a)
    time.append(t)

for i, a in enumerate(accuracy[1:]):
    for j, b in enumerate(a):
        accuracy[0][j] += accuracy[i][j]
accuracy.append([])
for i, a in enumerate(accuracy[0]):
    accuracy[5].append(a / 5)

time.append([])
for i, a in enumerate(time[0]):
    time[5].append(a / 5)
total = 0
SD = []
for i in range(10):
    for j in range(5):
        total += math.pow(accuracy[5][i] - accuracy[j][i], 2)
    total = math.sqrt(total / 5)
    SD.append(total)
    total = 0
fig, ax = plt.subplots()
fig.suptitle('Accuracy', fontsize=20)
ax.plot(list(range(10)), accuracy[5])
fig.savefig("neuralnetwork_digit_A.png")

fig, ax = plt.subplots()
fig.suptitle('Standard Deviation', fontsize=20)
ax.plot(list(range(10)), SD)
fig.savefig("neuralnetwork_digit_SD.png")

fig, ax = plt.subplots()
fig.suptitle('Run Time', fontsize=20)
ax.plot(list(range(1, 11)), time[5])
fig.savefig("neuralnetwork_digit_T.png")

# %%
