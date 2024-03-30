from Layer import S_Layer, C_Layer
from DataProcessing import CheckArrayIndex, createData, plotData
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Neocognitron:
    def __init__(self, Layer: list):
        # Layer[i] = [S_layer, C_Layer]
        # a.shape = (layer, planeNum, prevPlaneNum, receptiveField, receptiveField)
        # b.shape = (layer, planeNum)
        self.Layer = Layer
        self.a = {0: []}
        self.b = {0: []}

        for layer in range(1, len(Layer)):
            planeNum = Layer[layer][0].planeNum
            prevPlaneNum = Layer[layer-1][1].planeNum
            receptiveField = Layer[layer][0].receptiveField
            self.a[layer] = np.random.rand(planeNum, prevPlaneNum, receptiveField, receptiveField)*0.1
            self.b[layer] = np.zeros((planeNum))

    def save(self):
        for i in range(1, len(self.Layer)):
            np.save(f"weights\\a{i}.npy", self.a[i])
            np.save(f"weights\\b{i}.npy", self.b[i])

    def load(self):
        for i in range(1, len(self.Layer)):
            self.a[i] = np.load(f"weights\\a{i}.npy")
            self.b[i] = np.load(f"weights\\b{i}.npy")

    def setInput(self, input):
        # input.shape = (scale, size, size)
        assert input.shape == self.Layer[0][1].cLayer.shape
        self.Layer[0][1].cLayer = input 

    def forward(self, input):
        self.setInput(input)
        for layer in range(1, len(self.Layer)):
            self.Layer[layer][0].propagate(self.Layer[layer-1][1].cLayer, self.a[layer], self.b[layer])
            self.Layer[layer][1].propagate(self.Layer[layer][0].sLayer)

    def getCellLocation(self, layer, sColumnSize):
        residual = self.Layer[layer][0].planeSize - sColumnSize
        planeNum = self.Layer[layer][0].planeNum
        cellLocation = {i :[] for i in range(planeNum)}
        repPlane = 0
        repX = 0
        repY = 0
        for i in range(residual+1):
            for j in range(residual+1):
                maxVal = 0
                for plane in range(planeNum):
                    for x in range(sColumnSize):
                        for y in range(sColumnSize):
                            if self.Layer[layer][0].sLayer[plane][x+i][y+j] > maxVal:
                                maxVal = self.Layer[layer][0].sLayer[plane][x+i][y+j]
                                repPlane = plane
                                repX = x+i
                                repY = y+j
                cellLocation[repPlane].append([repX, repY])
        return cellLocation
    
    def getMax(self, layer, plane, repPlane):
        x = 0
        y = 0
        maxVal = 0
        for location in repPlane:
            if self.Layer[layer][0].sLayer[plane][location[0]][location[1]] > maxVal:
                x = location[0]
                y = location[1]
                maxVal = self.Layer[layer][0].sLayer[plane][x][y]
        return [x,y]
    
    
    def getReprsentativeCells(self, layer, sColumnSize):
        cellLocation = self.getCellLocation(layer, sColumnSize)
        repCells = []
        for plane in range(len(cellLocation)):
            if len(cellLocation[plane]) != 0: location = self.getMax(layer, plane, cellLocation[plane])
            else: location = None
            repCells.append(location)
        return repCells

    def update(self, layer, repCells, q):
        prevPlaneNum = self.Layer[layer-1][1].planeNum
        prevPlaneSize = self.Layer[layer-1][1].planeSize
        receptiveField = self.Layer[layer][0].receptiveField
        dimDiff = abs(prevPlaneSize - self.Layer[layer][0].planeSize)//2
        rf = receptiveField//2

        for plane, location in enumerate(repCells):
            if location != None:
                x = location[0]
                y = location[1]
                self.b[layer][plane] += q * self.Layer[layer][0].v[x][y]
                for prevPlane in range(prevPlaneNum):
                    for i in range(-rf, rf+1):
                        for j in range(-rf ,rf+1):
                            if CheckArrayIndex(x, y, i, j, dimDiff, prevPlaneSize):
                                prev = self.Layer[layer-1][1].cLayer[prevPlane][x+i+dimDiff][y+j+dimDiff]
                                self.a[layer][plane][prevPlane][i+rf][j+rf] += q*prev*self.Layer[layer][0].c[i+rf][j+rf]

    def train(self, inputs, sColumn, q, epochs):
        for epoch in range(epochs):
            print(f"\rEpoch {epoch+1}/{epochs}:\t" + "[" + "="*(10*0) + ">" + "-"*(50-10*0) + "]", end="")        
            for i, input in enumerate(inputs):          
                self.forward(input)
                for layer in range(1, len(self.Layer)):
                    repCells = self.getReprsentativeCells(layer, sColumn[layer])
                    self.update(layer, repCells, q[layer])
                print(f"\rEpoch {epoch+1}/{epochs}:\t" + "[" + "="*int((i+1)*5) + ">" + "-"*int(50-(i+1)*5) + "]", end="")
            print()

def main():
    pass

if __name__ == "__main__":
    main()
