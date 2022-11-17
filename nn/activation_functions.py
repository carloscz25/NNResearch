from math import exp, sinh, cosh

"""
Functions on this file are implemented according to explanations in the following url

https://machinelearningmastery.com/a-gentle-introduction-to-sigmoid-function/

https://functions.wolfram.com/ElementaryFunctions/Tanh/introductions/Tanh/ShowAll.html

"""

SIGMOID = 'sigmoid'
RELU = 'relu'
LEAKY_RELU = 'leaky_relu'
TANH = 'tanh'


activationfunctions = {}
activationfunctionsderivates = {}

def sigmoid(x):
    return 1/(1+exp(-x))

def sigmoid_derivate(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    if (x >0):
        return x
    else:
        return 0

def relu_derivative(x):
    if (x >0):
        return 1
    else:
        return 0

def leaky_relu(x):
    if (x >0):
        return x
    else:
        return -0.01*x

def leaky_relu_derivative(x):
    if (x >0):
        return 1
    else:
        return 0.01

def tanh(x):
    return (exp(x)-exp(-x))/(exp(x)+exp(-x))

def tanh_derivative(x):
    return (1-pow(tanh(x),2))

activationfunctions[SIGMOID] = sigmoid
activationfunctionsderivates[SIGMOID] = sigmoid_derivate
activationfunctions[RELU] = relu
activationfunctionsderivates[RELU] = relu_derivative
activationfunctions[LEAKY_RELU] = leaky_relu
activationfunctionsderivates[LEAKY_RELU] = leaky_relu_derivative
activationfunctions[TANH] = tanh
activationfunctionsderivates[TANH] = tanh_derivative

def tanh(x):
    return sinh(x)/cosh(x)