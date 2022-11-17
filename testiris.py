import numpy as np
from PySide6.QtWidgets import QApplication

from nn.basics import gradients, weights

# def sigmoid(x):
#     sig = 1 / (1 + math.exp(-x))
#     return sig


from ui.valueholder import ValueHolder
from nn.optimizers import SerenityTracker

errorholder = ValueHolder()


def nncode(errorholder):
    from dataloading.dataloading import irisproblem_datasetgenerator
    from nn.error_functions import errorfunctions, MEAN_SQUARED_ERROR, CROSS_ENTROPY_LOSS
    from nn.basics import Net
    from nn.optimizers import StochasticGradientDescentOptimizer, AdamOptimizer
    from nn.storage import StoredNetInfo
    from nn.accesory_functions import onehotencode


    net = Net.arrangeFullyConnectedForwardNet([4,24,12,3])
    net.applySoftmax = True
    net.calculatelinkandneuronpathsforgradientcomputation()
    net.init_weights(0.5,0.5)

    sni = StoredNetInfo('netinfo.nfo', net)

    # opt = StochasticGradientDescentOptimizer(learningrate=0.02, net=net,
    #                                          gradientreductionmethod='avg')
    # opt.gradienttracker = SerenityTracker()
    opt = AdamOptimizer(net=net, beta1=0.9, beta2=0.999, stepsize=0.005)
    cumerror = 0
    for i in range(100000):
        datasetgenerator = irisproblem_datasetgenerator(batchsize=64)
        res = []
        erroravg = []
        for data, label in datasetgenerator:


            res = net.forwardpass(data)
            error = errorfunctions[CROSS_ENTROPY_LOSS](label, res)
            erroravg.append(error)

            net.computegradientsperbackpropagation_withsoftmaxlayer(errorfunctionname=CROSS_ENTROPY_LOSS, target=label, x=res)

            net.resetnetfornewforwardpass()
            y = 2

        epoch, weights, grads = sni.copyEpochInfo(i)
        sni.appendEpochInfo(epoch, weights, grads, error)
        opt.adjustweights()

        grads = gradients(net)
        grads2 = []
        for j in range(5):
            g = grads[j]
            grads2.append(np.average(g))
        errorholder.value = grads2
        errorholder.value.append(np.average(erroravg))

        net.resetgradients()
        print("===========================EPOCH NUM. " + str(i) + "===========================")
        print(str(np.average(erroravg)))

        errorholder.epoch = i
        erroravg.clear()






import threading
import sys
import time
from ui.timeseriespanel import Chart


th = threading.Thread(target=nncode, args=[errorholder])


th.start()

time.sleep(5)

app = QApplication()

mw = Chart.createTimeSeriesPanelWithData(errorholder)

mw.show()


app.exec()














