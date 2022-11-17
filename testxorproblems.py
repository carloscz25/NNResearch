import numpy as np




# def sigmoid(x):
#     sig = 1 / (1 + math.exp(-x))
#     return sig





        





    






from ui.valueholder import ValueHolder

errorholder = ValueHolder()

def nncode(errorholder):
    from dataloading.dataloading import xorproblem_datasetgenerator
    from nn.error_functions import errorfunctions, MEAN_SQUARED_ERROR
    from nn.basics import Net
    from nn.optimizers import StochasticGradientDescentOptimizer, AdamOptimizer




    net = Net.arrangeFullyConnectedForwardNet([2,4,2,1])
    net.calculatelinkandneuronpathsforgradientcomputation()
    net.init_weights()

    opt = StochasticGradientDescentOptimizer(learningrate=0.9, net=net)
    # opt = AdamOptimizer(net=net, beta1=0.9, beta2=0.999,stepsize=0.001)
    cumerror = 0
    for i in range(1000000):
        datasetgenerator = xorproblem_datasetgenerator()
        res = []
        erroravg = []
        for data, label in datasetgenerator:
            res = net.forwardpass(data)[0]
            error = errorfunctions[MEAN_SQUARED_ERROR](label, res)
            erroravg.append(error)

            net.computegradientsperbackpropagation(errorfunctionname=MEAN_SQUARED_ERROR, target=label, x=res)

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

