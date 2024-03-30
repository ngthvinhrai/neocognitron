import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def CheckArrayIndex(x, y, rfi, rfj, dimDiff, prevPlaneSize):
    if x+rfi+dimDiff >= 0 and x+rfi+dimDiff < prevPlaneSize and y+rfj+dimDiff >= 0 and y+rfj+dimDiff < prevPlaneSize: 
        return True
    return False


def createData(path, numClass, numEachClass, scale, shape):
    # path: Path of entire dataset
    # numClass: Number of data class
    # numEachClass: Number of data each class
    # scale: 1D, 2D, 3D
    # shape: Output data shape

    data = np.zeros((numClass, numEachClass, scale, shape, shape))
    for i in range(numClass):
        dataList = os.listdir(os.path.join(path, f"{i}"))
        for j in range(numEachClass):
            img = cv2.imread(os.path.join(path, f"{i}", dataList[j]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (shape, shape))
            img = img/255.
            data[i][j][0:scale] = img 

    return data

def plotData(data, figsize, numRow, numCol):
    fig, axes = plt.subplots(numRow, numCol, figsize = (figsize,figsize))
    axes = axes.flatten()
    for i in range(data.shape[0]):
        axes[i].imshow(data[i])
        axes[i].axis("off")
    plt.show()    

def main():
    path = "E:\\Project\\neocognitron\\dataset\\mnist_digit_png\\mnist_png\\training"
    data = createData(path, 10, 1, 1, 16)
    plotData(data, 10, 2, 5)
if __name__ == "__main__":
    main()
