from nn.basics import Link, Neuron, Net
import numpy as np
import pickle
import codecs


class StoredNetInfo():

    def __init__(self, path, net):
        self.net = net
        self.path = path
        self.weights = {}
        self.gradients = {}
        self.epochs = {}
        f = open(self.path, 'w')
        pickled = self.encodeObjectToLineForStorage(self.net)
        f.write(pickled)
        f.close()

    def encodeObjectToLineForStorage(self, obj):
        pickled = codecs.encode(pickle.dumps(obj), "base64").decode().replace("\n", "")+"\n"
        return pickled

    def decodeLineToObject(self, line):
        obj = pickle.loads(codecs.decode(line.encode(), "base64"))
        return obj

    def copyEpochInfo(self, epoch):
        weights = {}
        l: Link = None
        n: Neuron = None
        for l  in self.net.all_links:
            weights['w-'+str(l.id)] = l.weight
        for n  in self.net.all_neurons():
            weights['b-'+str(n.id)] = n.bias
        gradients = {}
        l: Link = None
        n: Neuron = None
        for l in self.net.all_links:
            gradients['w-' + str(l.id)] = np.average(l.gradient)
        for n in self.net.all_neurons():
            gradients['b-' + str(n.id)] = np.average(n.biasgradient)

        return epoch, weights, gradients

    def appendEpochInfo(self, epoch : int = None, weights : dict = None, gradients : dict = None, error : float = None):
        f = open(self.path, 'a')
        ii = IterationInfo(epoch, weights, gradients, error)
        iistr = self.encodeObjectToLineForStorage(ii)
        f.write(iistr)
        f.close()


class IterationInfo():
    def __init__(self, epoch : int = None, weights : dict = None, gradients : dict = None, error : float = None):
        self.epoch = epoch
        self.weights = weights
        self.gradients = gradients
        self.error = error

