# defines a multilayer perceptron neural network class and associated algorithms

import numpy as np
import math


def sigmoid(x):
    """perform sigmoid function"""
    if x < 0:
        a = math.e**x
        return a / (1 + a)
    else:
        return 1 / (1 + math.e**(-x))

def arr_sigmoid(arr):
    """perform sigmoid on an array"""
    sig = []
    for f in arr:
        sig.append(sigmoid(f))
    return np.array(sig)

def inv_sigmoid(x):
    """perform inverse sigmoid function"""
    return -math.log(1/x - 1)

def inv_dif_sig(x):
    """perform inverse sigmoid function, then derivative of the sigmoid
    this operation simplifies significantly
    also handy as it works with array input"""
    return x - x**2

def random():
    """return a random float on the open interval (0, 1)"""
    x = np.random.random()
    if x == 0.0: # nasty edge case we don't want, negligible probability
        x = 0.5 # you saw nothing
    return x

def rand_array(x, y, func=lambda x:x):
    """generates an x by y array of random numbers
    with an inverse sigmoid distribution"""
    arr = []
    if y == 1:
        for f in range(x):
                arr.append(func(random()))
    else:
        for f in range(y):
            arr.append([])
            for g in range(x):
                arr[-1].append(func(random()))
    return np.array(arr)

def null_array(x, y, val=0):
    """generates an x by y array of zeros"""
    if y == 1:
        arr = x*[val]
    else:
        arr = []
        for f in range(y):
            arr.append(x*[val])
    return np.array(arr)

class network():

    def __init__(self, *layers):
        """initializes network with number of layers and neurons per layer
        specified in layers, where the first is input and the last is output"""
        self.layers = layers
        self.depth = len(layers)
        self.weights = []
        self.biases = []
        self.neurons = []
        for f in range(1, self.depth):
            self.weights.append(rand_array(self.layers[f-1], self.layers[f], func=inv_sigmoid))
            self.biases.append(rand_array(self.layers[f], 1, func=inv_sigmoid))
        for f in self.layers:
            self.neurons.append(np.array(f*[0]))

        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)



    def evaluate(self, data):
        """take input data and output the neural network's evaluation
        also leaves the network state in self.neurons"""
        if len(data) != self.layers[0]:
            raise ValueError("Network input of invalid length")
        self.neurons[0] = np.array(data)
        for f in range(self.depth-1):
            self.neurons[f+1] = arr_sigmoid((self.weights[f] @ self.neurons[f]) + self.biases[f])
        return self.neurons[-1]

    def cost(self, data, result):
        """claculate the distance between the evaluated result
        and the intended result"""
        actual = self.evaluate(data)
        result = np.reshape(result, self.layers[-1])
        return ((actual-result)**2).sum()

    def all_cost(self, data, result):
        """claculate the distance between the evaluated results
        and the intended results"""
        num = len(data)
        costs = []
        for f in range(num):
            costs.append(self.cost(data[f], result[f]))
        return costs

    def avg_cost(self, data, result):
        """claculate the average distance between the evaluated results
        and the intended results"""
        num = len(data)
        total_cost = 0
        for f in range(num):
            total_cost += self.cost(data[f], result[f])
        return total_cost/num

    def back_propagate(self, data_sets, result_sets, l_rate=1, momentum=[]):
        """back propagates the given data and intended results"""
        weight_change = []
        bias_change = []
        layer_change = []
        for f in range(self.depth-1):
            weight_change.append(null_array(self.layers[f], self.layers[f+1]))
            bias_change.append(null_array(self.layers[f+1], 1))
        for f in self.layers:
            layer_change.append(np.array(f*[0]))
        weight_change = np.array(weight_change)
        bias_change = np.array(bias_change)

        # for every data set, find the cost gradient and add to the total sum
        for cset in range(len(data_sets)):
            self.evaluate(data_sets[cset])
            layer_change[-1] = l_rate*(np.array(result_sets[cset])-self.neurons[-1])
            for f in range(self.depth-2, -1, -1):
                dz = inv_dif_sig(self.neurons[f+1])
                bias_change[f] = bias_change[f] + (dz*layer_change[f+1])
                weight_change[f] = weight_change[f] + (np.array([self.neurons[f]*dz[n]*layer_change[f+1][n] for n in range(len(self.neurons[f+1]))]))
                layer_change[f] = np.array([(self.weights[f][:,n]*dz*layer_change[f+1]).sum() for n in range(len(self.neurons[f]))])

        # add momentum if any
        if momentum != []:
            weight_change += momentum[0]
            bias_change += momentum[1]

        # add the total cost gradient to the current network
        self.weights += weight_change
        self.biases += bias_change

        # return the change this go around
        return np.array([weight_change, bias_change])
