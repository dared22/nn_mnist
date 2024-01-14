import torch
import torch.nn as nn
import torch.nn.functional as F

#simple nn for testing, will be replaced later

class CustomLayer(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(inputSize, outputSize))
        self.bias = nn.Parameter(torch.randn(outputSize))

    def forward(self, X):
        return torch.matmul(X, self.weights) + self.bias
    


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, n1=784, n2=32, n3=10):
        super().__init__()
        self.flatten = nn.Flatten()
        #input layer
        self.input = CustomLayer(n1, n2)
        #hidden layer
        self.layer = CustomLayer(n2,n2)
        #output layer
        self.output = CustomLayer(n2, n3)


    def forward(self, x):
            x = self.flatten(x) 
            x = F.relu(self.input(x))
            x = F.relu(self.layer(x))
            return torch.log_softmax(self.output(x), axis=1)

