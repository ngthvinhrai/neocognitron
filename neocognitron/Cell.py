import numpy as np
from DataProcessing import CheckArrayIndex

def relu(x):
    return max(0,x)

class sCell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cell = 0

    def __call__(self, input, receptiveField, a, b, v, rl, dimDiff):
        # input.shape = (prevPlaneNum, prevPlaneSize, prevPlaneSize)
        # a.shape = (prevPlaneNum, receptiveField, receptiveField)
        prevPlaneNum = input.shape[0]
        prevPlaneSize = input.shape[1]
        rf = receptiveField//2
        e = 0
        h = (rl/(1+rl))*b*v

        for plane in range(prevPlaneNum):
            for i in range(-rf, rf+1):
                for j in range(-rf, rf+1):
                    if CheckArrayIndex(self.x, self.y, i, j, dimDiff, prevPlaneSize):
                        prev = input[plane][self.x+i+dimDiff][self.y+j+dimDiff]
                        e += a[plane][i+rf][j+rf] * prev
        
        self.cell = rl * relu((1+e)/(1+h) - 1)
        return self.cell
    
class cCell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cell = 0

    def __call__(self, input, receptiveField, alpha, v, theta, subTheta, dimDiff):
        # input.shape = (prevPlaneSize, prevPlaneSize)
        prevPlaneSize = input.shape[1]
        rf = receptiveField//2
        e = 0

        for i in range(-rf, rf+1):
            for j in range(-rf, rf+1):
                dCell = theta*pow(subTheta, np.sqrt(i*i +j*j))
                if CheckArrayIndex(self.x, self.y, i, j, dimDiff, prevPlaneSize):
                    prev = input[self.x+i+dimDiff][self.y+j+dimDiff]
                    e += dCell*prev
        res = relu((1+e)/(1+v) - 1)
        self.cell = res/(res + alpha) 
        return self.cell
    
def main():
    pass

if __name__ == "__main__":
    main()
