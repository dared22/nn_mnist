import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DenseLayer(nn.Module):
    """
    A custom layer in PyTorch representing a dense layer with an optional activation function.
    
    input_size : int
        The size (number of features) of the input.
    output_size : int
        The size (number of features) of the output.
    activation : callable, optional
        The activation function to apply after the linear transformation.
        If None, no activation is applied. Default: None.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The weight matrix of the layer. Shape: (input_size, output_size). Initialized to random values.
    bias : torch.nn.Parameter
        The bias vector of the layer. Shape: (output_size,).
        Initialized to zeros.
    """

    def __init__(self, input_size, output_size, activation=None):
        super(DenseLayer, self).__init__()
        # initialize weights and bias to random values (not HE)
        self.weight = nn.Parameter(torch.randn(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.activation = activation

    def forward(self, x):
        """
        Computes the output of the DenseLayer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output of the layer.
        """
        x = torch.matmul(x, self.weight) + self.bias
        if self.activation:
            x = self.activation(x)
        return x