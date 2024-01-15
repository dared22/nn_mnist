import torch
import torch.nn as nn
import torch.nn.init as init

class VanillaLowRankLayer(nn.Module):
    """
    A custom PyTorch layer implementing a low-rank approximation of a dense layer.

    It utilizes three weight matrices to approximate dense layer transformations, 
    reducing parameter count. An optional activation function can be applied post-transformation.

    Parameters
    ----------
    input_size : int
        The size of input.
    output_size : int
        The size of output.
    rank : int
        The rank of the low rank approximation.
    activation : callable (optional)
        The activation function to apply after the linear transformation

    Attributes
    ----------
    weight1 : torch.nn.Parameter
        The first weight matrix of the layer with dimensions (input_size, rank).
        Initialized using He initialization.
    weight2 : torch.nn.Parameter
        The second weight matrix of the layer, a square matrix with dimensions (rank, rank).
        Initialized using He initialization.
    weight3 : torch.nn.Parameter
        The third weight matrix with dimensions (rank, output_size). 
        Initialized using He initialization.
    bias : torch.nn.Parameter
        The bias parameter of the layer, with the same number of elements as the output_size.
        Initialized with random values from a standard normal distribution.
    """

    def __init__(self, input_size, output_size, rank, activation=None):
        super(VanillaLowRankLayer, self).__init__()
        # Weight matricies
        self.weight1 = nn.Parameter(torch.Tensor(input_size, rank))
        self.weight2 = nn.Parameter(torch.Tensor(rank, rank))
        self.weight3 = nn.Parameter(torch.Tensor(rank, output_size))
        # Initialize the weights
        init.kaiming_normal_(self.weight1, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.weight2, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.weight3, mode='fan_in', nonlinearity='relu')
        # Bias
        self.bias = nn.Parameter(torch.randn(output_size))
        # Activation function
        self.activation = activation

    def forward(self, x):
        x = torch.matmul(x, self.weight1)
        x = torch.matmul(x, self.weight2)
        x = torch.matmul(x, self.weight3)
        x = x + self.bias # Add the bias
        if self.activation:
            x = self.activation(x)
        return x