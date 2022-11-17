import numpy as np

# def sigmoid(x):
#     sig = 1 / (1 + math.exp(-x))
#     return sig


from ui.valueholder import ValueHolder

errorholder = ValueHolder()


def nncode(errorholder):
    from dataloading.dataloading import generator_fashionmnist_dataset_train
    from nn.error_functions import errorfunctions, MEAN_SQUARED_ERROR, CROSS_ENTROPY_LOSS
    from nn.basics import Net
    from nn.optimizers import StochasticGradientDescentOptimizer, AdamOptimizer
    from nn.accesory_functions import onehotencode

    net = Net.arrangeFullyConnectedForwardNet([784, 128, 10])
    net.applySoftmax = True
    net.calculatelinkandneuronpathsforgradientcomputation()
    net.init_weights()

    opt = StochasticGradientDescentOptimizer(learningrate=0.03, net=net)
    # opt = AdamOptimizer(net=net, beta1=0.9, beta2=0.999, stepsize=0.001)
    cumerror = 0
    for i in range(100000):
        datasetgenerator = generator_fashionmnist_dataset_train()
        res = []
        erroravg = []
        for data, label in datasetgenerator:
            label1h = onehotencode(label,10)
            data = np.ndarray.flatten(data)
            data = data / 255
            res = net.forwardpass(data)
            error = errorfunctions[CROSS_ENTROPY_LOSS](label1h, res)
            erroravg.append(error)

            net.computegradientsperbackpropagation_withsoftmaxlayer(errorfunctionname=CROSS_ENTROPY_LOSS, target=label1h, x=res)

            net.resetnetfornewforwardpass()
            y = 2
        if (np.average(erroravg) < 0.001):
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

