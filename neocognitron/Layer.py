import numpy as np
from Cell import sCell, cCell
from VPlane import Vs, Vc
from DataProcessing import createData, plotData
import cv2

class Layer:
    def __init__(self, planeSize, planeNum, receptiveField):
        self.planeSize = planeSize
        self.planeNum = planeNum
        self.receptiveField = receptiveField

    # Propagate for S_Layer
    def propagate(self, input, a, b):
        pass

    # Propagate for C_Layer
    def propagate(self, input):
        pass

class S_Layer(Layer):
    def __init__(self, planeNum, planeSize, receptiveField, rl, gamma):
        super().__init__(planeSize, planeNum, receptiveField)
        self.rl = rl
        self.gamma = gamma
        self.sLayer = np.zeros((planeNum, planeSize, planeSize))
        self.v = np.zeros((planeSize, planeSize))
        self.c = np.zeros((receptiveField, receptiveField))

    def __generateC(self, Size, prevPlaneNum, gamma):
        sum = 0
        rf = Size//2
        for i in range(-rf, rf+1):
            for j in range(-rf, rf+1):
                self.c[i+rf][j+rf] = pow(gamma, np.sqrt(i*i + j*j))
                sum += self.c[i+rf][j+rf]

        for x in range(Size):
            for y in range(Size):
                self.c[x][y] = self.c[x][y]/(sum*prevPlaneNum)

    def propagate(self, input, a, b):
        # input.shape = (prevPlaneNum, prevPlaneSize, prevPlaneSize)
        # a.shape = (planeNum, prevPlanenum, receptiveField, receptiveField)
        # b.shape = (planeNum, )
        dimDiff = abs(input.shape[1] - self.planeSize)//2
        self.__generateC(self.receptiveField, input.shape[0], self.gamma)
        self.v = Vs(self.planeSize)(input, self.receptiveField, dimDiff, self.c)
        for plane in range(self.planeNum):
            for x in range(self.planeSize):
                for y in range(self.planeSize):
                    self.sLayer[plane][x][y] = sCell(x,y)(input, self.receptiveField, a[plane], b[plane], self.v[x][y], self.rl, dimDiff)



class C_Layer(Layer):
    def __init__(self, planeNum, planeSize, receptiveField, alpha, theta, subTheta):
        super().__init__(planeSize, planeNum, receptiveField)
        self.alpha = alpha
        self.theta = theta
        self.subTheta = subTheta
        self.cLayer = np.zeros((planeNum, planeSize, planeSize))
        self.v = np.zeros((planeSize, planeSize))

    def propagate(self, input):
        dimDiff = abs(input.shape[1] - self.planeSize)//2
        self.v = Vc(self.planeSize)(input, self.receptiveField, self.theta, self.subTheta, dimDiff)
        for plane in range(self.planeNum):
            for x in range(self.planeSize):
                for y in range(self.planeSize):
                    self.cLayer[plane][x][y] = cCell(x,y)(input[plane], self.receptiveField, self.alpha, self.v[x][y], self.theta, self.subTheta, dimDiff)




def main():
    pass

if __name__ == "__main__":
    main()