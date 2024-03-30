from Layer import S_Layer, C_Layer
from Neocognitron import Neocognitron
from DataProcessing import createData, plotData
import numpy as np
import os

def show():
    a = {0: []}
    b = {0: []}
    for i in range(1, 4):
        # a[i] = np.load(f"weights\\a{i}.npy")
        b[i] = np.load(f"weights\\b{i}.npy")
        print(b[i])


def main():
    path = "dataset\\mnist_digit_png\\mnist_png\\training"
    tPath = "dataset\\mnist_digit_png\\mnist_png\\testing"
    # data = createData(path, 5, 50, 1, 16)
    tData = createData(tPath, 10, 890, 1, 16)


    Layer =  [[S_Layer(0,0,0,0,0), C_Layer(1,16,0,0,0,0)],
                [S_Layer(24, 16, 5, 4, 0.9), C_Layer(24, 10, 5, 0.5, 4, 0.9)],
                [S_Layer(24, 8, 5, 1.5, 0.9), C_Layer(24, 6, 5, 0.5, 4, 0.8)],
                [S_Layer(24, 2, 5, 1.5, 0.9), C_Layer(24, 1, 2, 0.5, 2.5, 0.7)]]
    
    model = Neocognitron(Layer)
    if len(os.listdir("weights")):
        model.load()

    # for i in range(0,50):
    #     print(f"Dataset {i}:")
    #     model.train(data[:,i], sColumn=[0,5,5,2], q=[0, 0.2, 9.6, 13.94], epochs=1)
    #     model.save()

    #     for j, test in enumerate(tData):
    #         print(f"Number {j}: ", end="")
    #         for t in test:
    #             model.forward(t)
    #             print(np.argmax(model.Layer[3][1].cLayer), end=" ")
    #         print()

    # for i in range(5):
    #     print(f"Number {i}: ", end=" ")
    #     for j in range(10):
    #         model.forward(tData[i][j])
    #         print(np.argmax(model.Layer[3][1].cLayer), end=" ")
    #     print()
        
    for i in range(0,10):
        classes = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for j in range(tData.shape[1]):
            model.forward(tData[i][j])
            classes[np.argmax(model.Layer[3][1].cLayer)] += 1
            print(classes, end="\r")
        print()

    # plotData(model.a[1][:,0], 12, 2,19)

if __name__ == "__main__":
    main()
    # show()