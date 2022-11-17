from math import exp
from math import log, e
import numpy as np

MEAN_SQUARED_ERROR = 'mean_squared_error'
CROSS_ENTROPY_LOSS = 'cross_entropy_loss'

errorfunctions = {}
errorfunctionsderivates = {}


def meansquarederror(target, x):
    return pow((target-x),2)

def meansquarederror_derivate(target,x):
    return 2*(x-target)

def old_cross_entropy_loss(target, x):
    n = len(target)
    cum = 0
    for i in range(n):
        cum += (target[i]*log(x[i]))+((1-target[i])*log(1-x[i]))
    cel = (-1/n)*cum
    return cel

def old_cross_entropy_loss_derivate(target, x):
    target = np.array(target)
    x = np.array(x)
    celd = (target / x) + ((1-target)/(1-x))
    return celd

def cross_entropy_loss(target, x):
    n = len(target)
    cum = 0
    for i in range(n):
        try:
            cum += (target[i]*log(x[i], e))
        except BaseException as be:
            print(target)
            print(x)
            raise be
    cel = (-1)*cum
    return cel

def cross_entropy_loss_derivate(target, x):
    target = np.array(target)
    x = np.array(x)
    # celd = -target / x
    #celd = -1/x
    celd = (x - target)*2
    return celd

errorfunctions[MEAN_SQUARED_ERROR] = meansquarederror
errorfunctionsderivates[MEAN_SQUARED_ERROR] = meansquarederror_derivate

errorfunctions[CROSS_ENTROPY_LOSS] = cross_entropy_loss
errorfunctionsderivates[CROSS_ENTROPY_LOSS] = cross_entropy_loss_derivate


 
