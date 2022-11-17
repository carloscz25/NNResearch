import numpy as np
from math import sqrt



"""
functions implemented according to explanations in the following url

https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

"""

"""
FOR SIGMOID AND TANH ACTIVATION FUNCTIONS
"""

def xavier_weight_initialization(numer_of_nodes):
    n = numer_of_nodes
    lower, upper = -(1.0 / sqrt(n)), (1.0 / sqrt(n))
    w = np.random.uniform(lower, upper, numer_of_nodes)
    return w

def normalized_xavier_weight_initialization(numer_of_nodes_previous_layer, number_of_nodes_next_layer):
    n, m = numer_of_nodes_previous_layer, number_of_nodes_next_layer
    lower, upper = -(sqrt(6.0) / sqrt(n + m)), (sqrt(6.0) / sqrt(n + m))
    w = np.random.uniform(lower, upper, n)
    return w