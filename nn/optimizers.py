import numpy as np
from nn.basics import Net
import math

class Optimizer():

    def __init__(self):
        self.gradienttracker = None

    def adjustweights(self):
        pass

class GradientTracker():

    def __init__(self):
        pass

    def trackgradients(self, element, gradient):
        pass


class SerenityTracker(GradientTracker):
    """
    SerenitiTracker holds the number of sign switches during the whole training
    exercise, the final figure is obtained dividing the number of sign switches by the number
    of tracks, being a track each time the tracker is called.
    """
    def __init__(self):
        self.signswitches = {}
        self.lastgradients = {}
        self.numofcalls = 0

    def trackgradients(self, element, gradient):
        #initialization
        if not element in self.signswitches.keys():
            self.signswitches[element] = 0
        if not element in self.lastgradients.keys():
            self.lastgradients[element] = None
        self.numofcalls += 1

        lastgradient = self.lastgradients[element]
        if (lastgradient!=None):
            if checkoppositesign(lastgradient, gradient):
                self.signswitches[element] += 1
        self.lastgradients[element] = gradient

class StochasticGradientDescentOptimizer(Optimizer):

    def __init__(self, learningrate = 0.05, net: Net = None, clipgradients=None, gradientreductionmethod='avg'):
        super().__init__()
        self.learningrate = learningrate
        self.net = net
        self.clipgradients = clipgradients
        self.gradientreductionmethod = gradientreductionmethod

    def adjustweights(self):
        for l in self.net.all_links:
            if self.gradientreductionmethod == 'avg':
                gradientreduced = np.average(l.gradient)
            elif self.gradientreductionmethod == 'sum':
                gradientreduced = np.sum(l.gradient)
            else:
                raise BaseException("invalid reduction method")
            if self.gradienttracker != None:
                self.gradienttracker.trackgradients(l, gradientreduced)
            gradsum = gradientreduced
            if self.clipgradients!=None:
                gradsum = clipvalue(gradsum, self.clipgradients[0],self.clipgradients[1])
            learningstep = self.learningrate * gradsum
            l.weight -= learningstep

        flat_hidden = [item for sublist in self.net.hiddenneurons for item in sublist]
        for n in (self.net.inputneurons + flat_hidden + self.net.outputneurons):
            if self.gradientreductionmethod == 'avg':
                gradientreduced = np.average(l.gradient)
            elif self.gradientreductionmethod == 'sum':
                gradientreduced = np.sum(l.gradient)
            else:
                raise BaseException("invalid reduction method")
            if self.gradienttracker != None:
                self.gradienttracker.trackgradients(n, gradientreduced)
            gradsum = gradientreduced
            if self.clipgradients != None:
                gradsum = clipvalue(gradsum, self.clipgradients[0], self.clipgradients[1])
            n.bias -= self.learningrate * gradsum


class AdamOptimizer(Optimizer):

    def __init__(self, stepsize = 0.001, beta1 = 0.9, beta2 = 0.999, net=None):
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m0 = {}
        self.v0 = {}
        self.net = net
        self.timestep = 0
        self.stabilizer = 1e-23
        self.__prepare_m0_v0__()

    def __prepare_m0_v0__(self):
        for l in self.net.all_links:
            self.m0[l] = 0
            self.v0[l] = 0
        for n in self.net.all_neurons():
            self.m0[n] = 0
            self.v0[n] = 0

    def adjustweights(self):
        self.timestep = self.timestep + 1
        for l in self.net.all_links:
            gradient = np.sum(l.gradient)
            m = self.beta1*self.m0[l] + ((1-self.beta1)*gradient)
            v = self.beta2*self.v0[l] + ((1-self.beta2)*np.power(gradient,2))
            m_hat = m / (1-np.power(self.beta1,self.timestep))
            v_hat = v / (1-np.power(self.beta2, self.timestep))
            learningstep = (self.stepsize*(m_hat/(np.sqrt(v_hat)+self.stabilizer)))
            l.weight = l.weight - learningstep
            self.m0[l] = m
            self.v0[l] = v


        flat_hidden = [item for sublist in self.net.hiddenneurons for item in sublist]
        for n in (flat_hidden + self.net.outputneurons):
            gradient = np.sum(n.biasgradient)
            m = self.beta1 * self.m0[l] + ((1 - self.beta1) * gradient)
            v = self.beta2 * self.v0[l] + ((1 - self.beta2) * np.power(gradient, 2))
            m_hat = m / (1 - np.power(self.beta1, self.timestep))
            v_hat = v / (1 - np.power(self.beta2, self.timestep))
            learningstep = (self.stepsize * (m_hat / (np.sqrt(v_hat) + self.stabilizer)))
            n.bias = n.bias - learningstep
            self.m0[n] = m
            self.v0[n] = v

def checkoppositesign(a, b):
    if abs(a+b) != abs(a)+abs(b):
        return True
    else:
        return False

def clipvalue(val, lower, upper):
    if val < lower:
        return lower
    if val > upper:
        return upper
    else:
        return val


