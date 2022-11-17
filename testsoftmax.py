import numpy as np
from nn.basics import gradients, weights

# def sigmoid(x):
#     sig = 1 / (1 + math.exp(-x))
#     return sig


from ui.valueholder import ValueHolder
from nn.optimizers import SerenityTracker

errorholder = ValueHolder()


def nncode(errorholder):
    from dataloading.dataloading import generator_test_softmaxmodels
    from nn.error_functions import errorfunctions, MEAN_SQUARED_ERROR, CROSS_ENTROPY_LOSS
    from nn.basics import Net
    from nn.optimizers import StochasticGradientDescentOptimizer, AdamOptimizer
    from nn.accesory_functions import onehotencode

    net = Net.arrangeFullyConnectedForwardNet([2,20,2])
    net.applySoftmax = True
    net.calculatelinkandneuronpathsforgradientcomputation()
    net.init_weights(-2,2)

    opt = StochasticGradientDescentOptimizer(learningrate=0.03, net=net, clipgradients=[-1,1])
    opt.gradienttracker = SerenityTracker()
    # opt = AdamOptimizer(net=net, beta1=0.9, beta2=0.999, stepsize=0.0001)
    cumerror = 0
    for i in range(100000):
        datasetgenerator = generator_test_softmaxmodels()
        res = []
        erroravg = []
        for data, label in datasetgenerator:


            res = net.forwardpass(data)
            error = errorfunctions[CROSS_ENTROPY_LOSS](label, res)
            erroravg.append(error)

            net.computegradientsperbackpropagation_withsoftmaxlayer(errorfunctionname=CROSS_ENTROPY_LOSS, target=label, x=res)

            net.resetnetfornewforwardpass()
            y = 2

        opt.adjustweights()

        net.resetgradients()
        print("===========================EPOCH NUM. " + str(i) + "===========================")
        print(str(np.average(erroravg)))
        errorholder.value = np.average(erroravg)
        errorholder.epoch = i
        erroravg.clear()


def uicode(errorholder):
    from ui.timeseriespanel import runexample
    runexample(errorholder)


import threading

th = threading.Thread(target=nncode, args=[errorholder])
th2 = threading.Thread(target=uicode, args=[errorholder])

th.start()
th2.start()

y = 2

