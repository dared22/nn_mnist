import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super.(NeuralNetwork, self).__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
    
