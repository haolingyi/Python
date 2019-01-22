import numpy as np
import scipy.special


class SampleNeuralNetwork:
    def __init__(self, inpNum, hidNum, outNum, learnRate):
        self.inpNum = inpNum
        self.hidNum = hidNum
        self.outNum = outNum
        self.learnRate = learnRate
        self.wIH = np.random.normal(0.0, pow(self.hidNum, -0.5),
                                    [hidNum, inpNum])
        self.wHO = np.random.normal(0.0, pow(self.outNum, -0.5),
                                    [outNum, hidNum])

    def activation(self, inp):
        return scipy.special.expit(inp)

    def train(self, inpList, tragetList):

        inpOut = np.array(inpList, ndmin=2).T
        outTra = np.array(tragetList, ndmin=2).T

        hidInp = np.dot(self.wIH, inpOut)
        hidOut = self.activation(hidInp)
        outInp = np.dot(self.wHO, hidOut)
        outOut = self.activation(outInp)

        outErr = outTra - outOut
        hidErr = np.dot(self.wHO.T, outErr)

        self.wHO += self.learnRate * np.dot(
            (outErr * outOut * (1 - outOut)), np.transpose(hidOut))
        self.wIH += self.learnRate * np.dot(
            (hidErr * hidOut * (1 - hidOut)), np.transpose(inpOut))

    def query(self, inpList):
        inpOut = np.reshape(inpList, [self.inpNum, 1])
        hidInp = np.dot(self.wIH, inpOut)
        hidOut = self.activation(hidInp)
        outInp = np.dot(self.wHO, hidOut)
        outOut = self.activation(outInp)
        return outOut
