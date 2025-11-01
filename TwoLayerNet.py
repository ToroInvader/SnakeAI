import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict

class Affine:
    
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout): # is the current gradient due to chain rule
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx

class Relu:
    
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class SoftmaxWithLoss:

    def __init__(self):
        self.loss = None # Loss
        self.y = None # Output of softmax
        self.t = None # Label data (one-hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

class SigmoidWithLoss:

    def __init__(self):
        self.loss = None # Loss
        self.y = None # output of sigmoid
        self.t = None # target data

    def forward(self, x, t):
        self.t = t
        self.y = sigmoid(x)
        self.loss = sum_squared_error(self.y, self.t)  # quadratic loss
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # derivative of quadratic bit
        dx = self.y-self.t 
        dx *= ((1.0-self.y)*self.y) / batch_size
        return dx

class QuadraticLoss:

    def __init__(self):
        self.loss = None # Loss
        self.y = None # output of sigmoid
        self.t = None # target data

    def forward(self, x, t):
        self.t = t
        self.y = x
        self.loss = sum_squared_error(self.y, self.t)  # quadratic loss
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # derivative of quadratic bit
        dx = (self.y-self.t) /batch_size
        return dx


class Sigmoid:

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out
        return dx

class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #Initialise weights and biases
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # Create layers
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = QuadraticLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x: input data, t: label data
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    # x: input data, t: teacher data
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])
        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values()) # i didn't know you can cast list data types
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Settings
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W2"], grads["b2"] = self.layers["Affine2"].dW, self.layers["Affine2"].db
        
        return grads


def sigmoid(x):
    return 1 / 1+(np.exp(-x))

def identity_function(x):  
    return x

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape
    return -np.sum(t * np.log(y+delta))

def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()   
        
    return grad
