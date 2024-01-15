import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DenseLayer(nn.Module):
    """
    A custom layer in PyTorch representing a dense layer with an optional activation function.
    
    Parameters
    ----------
    input_size : int
        The size of the input.
    output_size : int
        The size of the output.
    activation : callable (optional)
        The activation function to apply after the linear transformation.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The weight matrix for the layer. Initialized using He initialization.
    bias : torch.nn.Parameter
        The bias parameter of the layer. Initialized with zeros.
    """

    def __init__(self, input_size, output_size, activation=None):
        super(DenseLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.activation = activation

    def forward(self, x):
        x = torch.matmul(x, self.weight) + self.bias
        if self.activation:
            x = self.activation(x)
        return x