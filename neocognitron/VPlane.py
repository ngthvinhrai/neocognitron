import numpy as np
from DataProcessing import CheckArrayIndex

# Vs == V_cl-1 in original paper
class Vs:
    def __init__(self, planeSize):
        self.planeSize = planeSize
        self.v = np.zeros((planeSize, planeSize))

    def __call__(self, input, receptiveField, dimDiff, c):
        prevPlaneNum = input.shape[0]
        prevPlaneSize = input.shape[1]
        rf = receptiveField//2

        for x in range(self.planeSize):
            for y in range(self.planeSize):
                for plane in range(prevPlaneNum):
                    for i in range(-rf, rf+1):
                        for j in range(-rf, rf+1):
                            cCell = c[i+rf][j+rf]
                            if CheckArrayIndex(x, y, i, j, dimDiff, prevPlaneSize):
                                prev = input[plane][x+i+dimDiff][y+j+dimDiff]
                                self.v[x][y] += cCell * prev**2
                self.v[x][y] = np.sqrt(self.v[x][y])
        
        return self.v

# Vc == V_sl in original paper    
class Vc:
    def __init__(self, planeSize):
        self.planeSize = planeSize
        self.v = np.zeros((planeSize, planeSize))

    def __call__(self, input, receptiveField, theta, subTheta, dimDiff):
        prevPlaneNum = input.shape[0]
        prevPlaneSize = input.shape[1]
        rf = receptiveField//2
        
        for x in range(self.planeSize):
            for y in range(self.planeSize):
                for plane in range(prevPlaneNum):
                    for i in range(-rf, rf+1):
                        for j in range(-rf, rf+1):
                            dCell = theta*pow(subTheta, np.sqrt(i*i + j*j))
                            if CheckArrayIndex(x, y, i, j, dimDiff, prevPlaneSize):
                                prev = input[plane][x+i+dimDiff][y+j+dimDiff]
                                self.v[x][y] += dCell * prev
                self.v[x][y] = self.v[x][y]/prevPlaneNum
        return self.v

def main():
    pass

if __name__ == "__main__":
    main()