import SampleNeuralNetwork as Net
import os
import matplotlib.pyplot as plt
import numpy as np
import time

InpNum = 784
HidNum = 200
OutNum = 10
LeaRate = 0.25
traNum = 60000
testNum = 10000


def showNumPic(imgNum, imgData):
    imgShow = np.asfarray(imgData).reshape(28, 28)
    print('Imge number is : %d' % int(imgNum[0]))
    plt.imshow(imgShow, cmap='Greys', interpolation='None')
    plt.show()
    plt.savefig('temp.jpg')


def train(net):
    trainFile = open('./data/train.csv')
    i = 0
    for eachPicSorData in trainFile:
        eachPicData = eachPicSorData.split(',')
        lableNum = int(eachPicData[0])
        imgData = eachPicData[1:]
        inplast = (np.asfarray(imgData) / 255.0 * 0.99) + 0.01
        tarlist = np.zeros(OutNum) + 0.01
        tarlist[lableNum] = 0.99
        net.train(inplast, tarlist)
        i += 1
        if (i % 1000 == 0):
            print('The training process : %.2f' % (i * 100.0 / traNum) + '%')


def test(net):
    tureNum = 0
    i = 0
    testFile = open('./data/test.csv')
    for eachPicSorData in testFile:
        eachPicData = eachPicSorData.split(',')
        lableNum = int(eachPicData[0])
        imgData = eachPicData[1:]
        inplast = (np.asfarray(imgData) / 255.0 * 0.99) + 0.01
        testRes = net.query(inplast)
        if np.argmax(testRes) == lableNum:
            tureNum += 1
        i += 1
        if (i % 1000 == 0):
            print('The testing process : %.2f' % (i * 100.0 / testNum) + '%')
    correct = tureNum * 100.0 / testNum
    print('correct:%.2f' % correct + '%')
    return correct


def saveRes(net):
    fileName = time.strftime("./result/%y-%m-%d %H:%M:%S Weight Result.npz",
                             time.localtime())
    np.savez(fileName, net.wIH, net.wHO)
    print('Save Result as Name :' + fileName)
    return fileName


def loadNet(fileName, net):
    r = np.load(fileName)
    print('Save Result from Name :' + fileName)
    net.wIH = r["arr_0"]
    net.wHO = r["arr_1"]


if __name__ == "__main__":
    x = []
    corret = []
    for i in range(1, 2):
        net = Net.SampleNeuralNetwork(InpNum, HidNum, OutNum, LeaRate * i)
        for j in range(0, 5):
            train(net)
        corret.append(test(net))
        x.append(LeaRate * i)
        saveRes(net)
    plt.plot(x, corret)
    plt.show()
    # input('Please Enter Any Keys To End')
