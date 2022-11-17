
import random
import numpy as np
from nn.activation_functions import activationfunctions, activationfunctionsderivates
from nn.error_functions import errorfunctions, errorfunctionsderivates
from nn.accesory_functions import softmax, softmax_derivate
from uuid import uuid4
import pickle

TYPE_OUTPUT = "OUTPUT"
TYPE_INPUT = "INPUT"
TYPE_HIDDEN = "HIDDEN"

INIT_WEIGHT_RANDOM = "IW_RANDOM"

class Neuron():
    def __init__(self, type):
        from nn.activation_functions import SIGMOID, RELU, TANH, LEAKY_RELU
        self.neurontype = type
        self.inputlinks = []
        self.activationfunctionname = LEAKY_RELU
        self.outputlinks = []
        self.id = uuid4()
        # forwardpass value holder
        self.fp_value = None
        self.net_value = None
        self.bias = random.uniform(-1, 1)
        self.biasgradient = []
        self.pathsforgradientcomputation = []




    @staticmethod
    def create_array_of_neurons(length, type):
        arr = [Neuron(type) for i in range(length)]
        return arr

    def gen_inputlinks(self):
        for self.idx, l in enumerate(self.inputlinks):
            yield l

class Link():
    def __init__(self, neuronfrom, neuronto):
        self.id = uuid4()
        self.neuronfrom = neuronfrom
        self.neuronto = neuronto
        self.weight = 0.0
        self.neuronfrom.outputlinks.append(self)
        self.neuronto.inputlinks.append(self)
        #forwardpass value holder
        self.fp_invalue = None
        self.fp_outvalue = None
        #gradient value holder for backpass
        self.gradient = []
        self.pathsforgradientcomputation = []


class Net():
    def __init__(self, applySoftmax : bool = False):
        self.inputneurons = []
        self.outputneurons = []
        self.hiddenneurons = []
        self.all_links = []
        self.init_weight_policy = INIT_WEIGHT_RANDOM
        self.applySoftmax = applySoftmax
        if (len(self.outputneurons)<2) and (applySoftmax==True):
            raise BaseException("Can't create net with softmax layer and less than 2 output neurons")


    def all_neurons(self):
        flat_hidden = [item for sublist in self.hiddenneurons for item in sublist]
        return self.inputneurons + flat_hidden + self.outputneurons;

    @staticmethod
    def arrangeFullyConnectedForwardNet(disposition: [] = None):
        net = Net()
        net.inputneurons = []
        net.outputneurons = []
        net.hiddenneurons = []
        net.all_links = []
        net.init_weight_policy = INIT_WEIGHT_RANDOM

        net.inputneurons = [Neuron(TYPE_INPUT) for i in range(disposition[0])]
        net.hiddenneurons = []
        for i in range(1, len(disposition) - 1):
            net.hiddenneurons.append([Neuron(TYPE_HIDDEN) for i in range(disposition[i])])

        net.outputneurons = [Neuron(TYPE_OUTPUT) for i in range(disposition[len(disposition) - 1])]
        net.all_links = []
        # creating links
        for i in net.inputneurons:
            for j in net.hiddenneurons[0]:
                l = Link(i, j)
                net.all_links.append(l)
        for i in range(len(net.hiddenneurons) - 1):
            for j in range(len(net.hiddenneurons[i])):
                for k in range(len(net.hiddenneurons[i + 1])):
                    l = Link(net.hiddenneurons[i][j], net.hiddenneurons[i + 1][k])
                    net.all_links.append(l)

        lasthiddenlayer_ind = len(net.hiddenneurons)-1
        for i in range(len(net.hiddenneurons[lasthiddenlayer_ind])):
            for j in range(len(net.outputneurons)):
                l = Link(net.hiddenneurons[lasthiddenlayer_ind][i], net.outputneurons[j]);
                net.all_links.append(l)
        return net

    def calculatelinkandneuronpathsforgradientcomputation(self):
        currpath = []
        on: Neuron = None
        for on in self.outputneurons:
            currneuron = on
            currlinkgenerator = currneuron.gen_inputlinks()
            currpath.append(currneuron)
            currneuron.pathsforgradientcomputation.append(currpath.copy())

            neuronsbuffer = []
            generatorbuffer = []
            while ((currneuron != None) and (currlinkgenerator != None)):
                l: Link = None
                try:
                    l = next(currlinkgenerator)
                except:
                    l = None
                if l != None:
                    currpath.append(l)
                    l.pathsforgradientcomputation.append(currpath.copy())
                    if (l.neuronfrom != None):
                        neuronsbuffer.append(currneuron)
                        generatorbuffer.append(currlinkgenerator)
                        currneuron = l.neuronfrom
                        currpath.append(currneuron)
                        currlinkgenerator = currneuron.gen_inputlinks()
                        currneuron.pathsforgradientcomputation.append(currpath.copy())
                else:
                    if len(neuronsbuffer) > 0:
                        currneuron = neuronsbuffer.pop()
                        currlinkgenerator = generatorbuffer.pop()

                    else:
                        currneuron = None
                        currlinkgenerator = None
                    currpath.pop()
                    if currneuron != None:
                        currpath.pop()

    def resetnetfornewforwardpass(self):
        for l in self.all_links:
            l.fp_invalue = None
            l.fp_outvalue = None

        for n in self.inputneurons:
            n.fp_value = None
            n.net_value = None

        flat_hidden = [item for sublist in self.hiddenneurons for item in sublist]
        for n in flat_hidden:
            n.fp_value = None
            n.net_value = None
        for n in self.outputneurons:
            n.fp_value = None
            n.net_value = None

    def resetgradients(self):
        for l in self.all_links:
            l.gradient = []

        for n in self.inputneurons:
            n.biasgradient = []

        flat_hidden = [item for sublist in self.hiddenneurons for item in sublist]
        for n in flat_hidden:
            n.biasgradient = []

        for n in self.outputneurons:
            n.biasgradient = []

    def init_weights(self, fromru=-0.3, toru=0.3):
        links = self.all_links
        for l in links:
            l.weight = random.uniform(fromru, toru)

    def serialize_weights(self, outputpath):
        weights = {}
        for l in self.all_links:
            weights[l.id] = l.weight
        for n in self.all_neurons():
            weights[n.id] = n.bias
        outfile = open(outputpath, 'wb')
        pickle.dump(weights, outfile)
        outfile.close()

    def loadweights(self, inputpath):
        infile = open(inputpath, 'rb')
        dict = pickle.load(inputpath)
        


    def forwardpass(self, data):
        if len(data) != len(self.inputneurons):
            raise BaseException("daTa input and net input dimension need to match")
        for i in range(len(data)):
            self.inputneurons[i].fp_value = data[i]
        pendinglinks, completedlinks = self.all_links.copy(), []
        flat_hidden = [item for sublist in self.hiddenneurons for item in sublist]
        pendingneurons, completedneurons = (flat_hidden + self.outputneurons).copy(), []

        while ((len(pendinglinks) > 0) and (len(pendingneurons) > 0)):
            toremovelinks = []
            for l in pendinglinks:
                if l.neuronfrom.fp_value != None:
                    l.fp_invalue = l.neuronfrom.fp_value
                    l.fp_outvalue = l.fp_invalue * l.weight
                    toremovelinks.append(l)
            for l in toremovelinks:
                pendinglinks.remove(l)
                completedlinks.append(l)

            toremoveneurons = []
            n: Neuron = None
            for n in pendingneurons:
                processneuron = True
                for l in n.inputlinks:
                    if l.fp_outvalue == None:
                        processneuron = False
                        break
                if processneuron == True:
                    cum = 0
                    for l in n.inputlinks:
                        cum = cum + l.fp_outvalue
                    cum += n.bias
                    n.net_value = cum
                    n.fp_value = activationfunctions[n.activationfunctionname](n.net_value)
                    toremoveneurons.append(n)
            for n in toremoveneurons:
                pendingneurons.remove(n)
                completedneurons.append(n)

        # returning an array with results
        resdata = []
        for n in self.outputneurons:
            resdata.append(n.fp_value)
        if self.applySoftmax:
            resdata = softmax(resdata)

        return resdata

    def computegradientsperbackpropagation_withsoftmaxlayer(self, errorfunctionname, target, x):

        errorfunctiondeltarulecomponent = errorfunctionsderivates[errorfunctionname](target, x)
        jacobian_softmax = softmax_derivate(x)
        on: Neuron = None
        for l in self.all_links:
            paths = l.pathsforgradientcomputation
            gradient = 0
            for p in paths:
                pathgrad = 1
                indexOfOutputNeuron = -1
                for el in reversed(p):
                    if (type(el) == Neuron):
                        neuron: Neuron = el
                        # pathgrad = pathgrad * (neuron.fp_value * (1 - neuron.fp_value))
                        pathgrad = pathgrad * activationfunctionsderivates[neuron.activationfunctionname](neuron.net_value)
                        if neuron.neurontype == TYPE_OUTPUT:
                            indexOfOutputNeuron = self.outputneurons.index(neuron)
                    elif (type(el) == Link):
                        link: Link = el
                        pathgrad = pathgrad * link.fp_invalue

                pathgrad = pathgrad
                pathgrad = pathgrad * jacobian_softmax[indexOfOutputNeuron][indexOfOutputNeuron]
                pathgrad = pathgrad * errorfunctiondeltarulecomponent[indexOfOutputNeuron]
                gradient = gradient + pathgrad
            l.gradient.append(gradient)


        n: Neuron = None
        for n in self.all_neurons():
            if n.neurontype == TYPE_INPUT:
                continue
            paths = n.pathsforgradientcomputation
            gradient = 0
            for p in paths:
                pathgrad = 1
                indexOfOutputNeuron = -1
                for el in p:
                    if (type(el) == Neuron):
                        neuron: Neuron = el
                        # pathgrad = pathgrad * (neuron.fp_value * (1 - neuron.fp_value))
                        pathgrad = pathgrad * activationfunctionsderivates[neuron.activationfunctionname](neuron.net_value)
                        if neuron.neurontype == TYPE_OUTPUT:
                            indexOfOutputNeuron = self.outputneurons.index(neuron)
                    elif (type(el) == Link):
                        link: Link = el
                        pathgrad = pathgrad * link.fp_invalue

                pathgrad = pathgrad
                pathgrad = pathgrad * jacobian_softmax[indexOfOutputNeuron][indexOfOutputNeuron]
                pathgrad = pathgrad * errorfunctiondeltarulecomponent[indexOfOutputNeuron]
                gradient = gradient + pathgrad
            n.biasgradient.append(gradient)

    def computegradientsperbackpropagation(self, errorfunctionname, target, x):

        errorfunctiondeltarulecomponent = errorfunctionsderivates[errorfunctionname](target, x)

        on: Neuron = None
        for l in self.all_links:
            paths = l.pathsforgradientcomputation
            gradient = 0
            for p in paths:
                pathgrad = 1
                indexOfOutputNeuron = -1
                for el in reversed(p):
                    if (type(el) == Neuron):
                        neuron: Neuron = el
                        # pathgrad = pathgrad * (neuron.fp_value * (1 - neuron.fp_value))
                        pathgrad = pathgrad * activationfunctionsderivates[neuron.activationfunctionname](
                            neuron.net_value)
                        if neuron.neurontype == TYPE_OUTPUT:
                            indexOfOutputNeuron = self.outputneurons.index(neuron)
                    elif (type(el) == Link):
                        link: Link = el
                        pathgrad = pathgrad * link.fp_invalue

                pathgrad = pathgrad
                gradient = gradient + pathgrad
            l.gradient.append(gradient * errorfunctiondeltarulecomponent)

        n: Neuron = None
        for n in self.all_neurons():
            if n.neurontype == TYPE_INPUT:
                continue
            paths = n.pathsforgradientcomputation
            gradient = 0
            for p in reversed(paths):
                pathgrad = 1
                for el in p:
                    if (type(el) == Neuron):
                        neuron: Neuron = el
                        # pathgrad = pathgrad * (neuron.fp_value * (1 - neuron.fp_value))
                        pathgrad = pathgrad * activationfunctionsderivates[neuron.activationfunctionname](
                            neuron.net_value)
                    elif (type(el) == Link):
                        link: Link = el
                        pathgrad = pathgrad * link.fp_invalue

                pathgrad = pathgrad
                gradient = gradient + pathgrad
            n.biasgradient.append(gradient * errorfunctiondeltarulecomponent)



def weights(net: Net = None):
    neurons = net.all_neurons()
    weights = []
    n : Neuron = None
    for n in neurons:
        ol : Link = None
        for ol in n.outputlinks:
            weights.append(ol.weight)
    return weights

def gradients(net : Net = None):
    neurons = net.all_neurons()
    gradients = []
    n : Neuron = None
    for n in neurons:
        ol : Link = None
        for ol in n.outputlinks:
            gradients.append(ol.gradient)
    return gradients

def gradients_avg(net : Net = None):
    neurons = net.all_neurons()
    gradients = []
    n : Neuron = None
    for n in neurons:
        ol : Link = None
        for ol in n.outputlinks:
            gradients.append(np.average(ol.gradient))
    return gradients
