import torch
import torch.nn as nn
import torch.nn.functional as F


A_F = ['relu', 'tanh', 'sigmoid'] 

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_function = 'relu'):
        super(NeuralNetwork, self).__init__()

    
        if activation_function not in A_F:
            raise ValueError("Invalid activation function. Supported activation functions: 'relu', 'tanh', 'sigmoid'")
        
        layers = []

        
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(self.get_activation(activation_function))

       
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(self.get_activation(activation_function))

       
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.Network = nn.Sequential(layers)

    
    def Forward(self, X):
        return self.Network(X)
    
    def GetActivation(self, activation_function):
        if activation_function == 'relu':
            return nn.ReLU()
        elif activation_function == 'tanh':
            return nn.Tanh()
        elif activation_function == 'sigmoid':
            return nn.Sigmoid()
        
    



