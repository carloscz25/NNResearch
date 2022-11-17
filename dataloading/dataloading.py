#to read ubyte files I followed instructions contained in following url:
#https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python

import idx2numpy
import paths
import os
import numpy as np
import random

from nn.accesory_functions import onehotencode


def generator_fashionmnist_dataset_train(size=4):
    file = os.path.join(paths.fashionmnist_dataset, "train-images-idx3-ubyte")
    images = idx2numpy.convert_from_file(file)

    file = os.path.join(paths.fashionmnist_dataset, "train-labels-idx1-ubyte")
    labels = idx2numpy.convert_from_file(file)

    datasetsize = len(images)

    for i in range(size):
        index = random.randrange(datasetsize)
        image = images[index]
        label = labels[index]
        yield image, label

def generator_test_softmaxmodels():
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [[0,0], [1,0], [0,1], [1,1]]
    i = 0

    for i in range(120):
        ind = random.randint(0,3)
        yield data[ind], labels[ind]

def xorproblem_datasetgenerator():
    data = [[0,0], [0,1], [1,0],[1,1]]
    labels = [0,1,1,0]
    i = 0
    while(i < len(labels)):
        yield data[i], labels[i]
        i = i + 1

def orproblem_datasetgenerator():
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 1, 1, 1]
    i = 0
    while (i < len(labels)):
        yield data[i], labels[i]
        i = i + 1

def irisproblem_datasetgenerator(batchsize = 32):
    labels_str = ["Iris-setosa","Iris-versicolor", "Iris-virginica"]
    path = os.path.join(paths.iris_dataset, "iris.data")
    f = open(path, 'r')
    lines = f.readlines()
    vals = []
    labels = []
    for i in range(len(lines)-1):
        l = lines[i]
        cols =l.split(",")
        vals.append([float(i) for i in cols[0:4]])
        labels.append(cols[4].replace("\n",""))
    vals = np.array(vals)
    labels = np.array(labels)

    vals = (vals - vals.mean(axis=0))/vals.std(axis=0)

    size = len(vals)
    for i in range(batchsize):
        index = random.randrange(size)
        val = vals[index]
        label = labels[index]
        label = onehotencode(labels_str.index(label), 3)
        yield val, label