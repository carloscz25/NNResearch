from math import exp
import numpy as np


def onehotencode(a :int = None, length:int = None):
    arr = [0 for i in range(length)]
    arr[a-1] = 1
    return arr

def onehotdecode(arr):
    return np.argmax(arr)

def softmax(x):
    try:
        cum = 0
        for i in x:
            cum += exp(i)
        xres = []
        for i in range(len(x)):
            xres.append(exp(x[i])/cum)
        return xres
    except:
        pass

def softmax_derivate(x):
    #return the jacobian of the softmax result : according to
    # result described in https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    jac = []
    for i in range(len(x)):
        jac.append([])
        for j in range(len(x)):
            if i==j:
                jac[i].append(x[i]*(1-x[i]))
            else:
                jac[i].append(-x[i]*x[j])
    jac = np.array(jac)
    return jac



