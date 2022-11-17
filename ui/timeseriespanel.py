from PySide6.QtCharts import QChart, QSplineSeries, QValueAxis, QChartView
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QPen, QIcon
import random
import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QHBoxLayout, QPushButton


class Chart(QChart):
    def __init__(self, parent=None, valueholder=None, title="Error Average"):
        super().__init__(QChart.ChartTypeCartesian, parent, Qt.WindowFlags())
        self._timer = QTimer()

        self.setTitle(title)
        self._titles = []
        self._axisX = QValueAxis()
        self._axisX.setTitleText("Epoch")
        self._axisY = QValueAxis()
        self.currrangeY = [0,0]
        self._step = 0
        self._x = 5
        self._y = 1

        self._timer.timeout.connect(self.handleTimeout)
        self._timer.setInterval(100)

        self.addAxis(self._axisX, Qt.AlignBottom)
        self.addAxis(self._axisY, Qt.AlignLeft)

        if type(valueholder.value)==list:
            self._series = []
            for i in range(len(valueholder.value)):
                self._series.append(QSplineSeries(self))
                seriesindex = len(self._series)-1

                self.addSeries(self._series[seriesindex])
                self._series[seriesindex].attachAxis(self._axisX)
                self._series[seriesindex].attachAxis(self._axisY)
                if (i == len(valueholder.value)-1):
                    green = QPen(Qt.green)
                else:
                    green = QPen(Qt.red)
                green.setWidth(1)
                self._series[seriesindex].setPen(green)
        else:
            self._series = QSplineSeries(self)
            green = QPen(Qt.red)
            green.setWidth(1)
            self._series.setPen(green)
            self.addSeries(self._series)
            self._series.attachAxis(self._axisX)
            self._series.attachAxis(self._axisY)






        self._axisX.setTickCount(5)
        self._axisX.setRange(0, 10)
        self._axisY.setRange(self.currrangeY[0], self.currrangeY[1])

        self._timer.start()
        self.valueholder = valueholder

    @Slot()
    def handleTimeout(self):
        if (self.valueholder.epoch!=None):
            maxval = 0
            if type(self.valueholder.value)!=list:
                self._series.append(self.valueholder.epoch, self.valueholder.value)
                if maxval<self.valueholder.value:
                    maxval = self.valueholder.value
                if (self.currrangeY[1] < self.valueholder.value):
                    self.currrangeY[1] = self.valueholder.value + 1
                if (self.currrangeY[0] > self.valueholder.value):
                    self.currrangeY[0] = self.valueholder.value - 1
            else:
                for i in range(len(self.valueholder.value)):
                    self._series[i].append(self.valueholder.epoch,self.valueholder.value[i])
                    if (self.currrangeY[1] < self.valueholder.value[i]):
                        self.currrangeY[1] = self.valueholder.value[i] + 1
                    if (self.currrangeY[0] > self.valueholder.value[i]):
                        self.currrangeY[0] = self.valueholder.value[i] - 1

            self._axisX.setRange(0, self.valueholder.epoch)


            self._axisY.setRange(self.currrangeY[0], self.currrangeY[1])

    @staticmethod
    def createTimeSeriesPanelWithData(valueholder):
        mw = MainWindow(valueholder)
        mw.show()
        return mw





class MainWindow(QMainWindow):

    def __init__(self, valueholder):
        super().__init__()
        self.bg = QFrame()

        self.resize(500, 500)
        self.setCentralWidget(self.bg)
        self.bg.layout = QVBoxLayout(self.bg)
        self.bg.layout.setSpacing(0)
        frame = self.gettopframe()

        self.bg.layout.addWidget(frame)
        chartview = QChartView(Chart(valueholder=valueholder))
        self.bg.layout.addWidget(chartview)
        self.update()

    def gettopframe(self):
        frame = QFrame()
        frame.layout = QHBoxLayout(frame)

        icon = QIcon("check.png")
        pushbutton = QPushButton(icon=icon)

        frame.layout.addWidget(pushbutton)
        return frame

from ui.valueholder import ValueHolder
vh = ValueHolder()
vh.epoch = 0
vh.value = 0.4
app = QApplication()
mw = MainWindow(vh)
mw.show()
app.exec()






