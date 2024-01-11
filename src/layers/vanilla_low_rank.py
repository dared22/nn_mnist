import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class VanillaLowRankLayer(nn.Module):
    """
    A custom layer in PyTorch representing a vanilla low rank dense layer.
    
    This layer performs matrix multiplication in succession with three different weight matrices.

    Parameters
    ----------
    input_size : int
        The size of input.
    output_size : int
        The size of output.
    rank : int
        The rank of the low rank approximation.

    Attributes
    ----------
    weight1 : torch.nn.Parameter
        The first weight matrix of the layer with dimensions (input_size, rank). 
        Initialized using He initialization suitable for ReLU activations.
    weight2 : torch.nn.Parameter
        The second weight matrix of the layer, a square matrix with dimensions (rank, rank).
        Also initialized using He initialization.
    weight3 : torch.nn.Parameter
        The third weight matrix with dimensions (rank, output_size). Also initialized using He initialization
    bias : torch.nn.Parameter
        The bias parameter of the layer, with the same number of elements as the 
        output_size. Initialized with random values from a standard normal distribution.
        """

    def __init__(self, input_size, output_size, rank):
        super(VanillaLowRankLayer, self).__init__()
        # Define the weight matrices
        self.weight1 = nn.Parameter(torch.Tensor(input_size, rank))
        self.weight2 = nn.Parameter(torch.Tensor(rank, rank))
        self.weight3 = nn.Parameter(torch.Tensor(rank, output_size))
        
        # Initialize the weights using He initialization (because we are mainly using ReLU)
        init.kaiming_normal_(self.weight1, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.weight2, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.weight3, mode='fan_in', nonlinearity='relu')
        
        # Initialize the bias with random values
        self.bias = nn.Parameter(torch.randn(output_size)) 
   
    def forward(self, x):
        """
        Defines the forward pass of the VanillaLowRankLayer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after applying matrix multiplications with the weight matrices.
        """
        x = torch.matmul(x, self.weight1)  # First matrix multiplication
        x = torch.matmul(x, self.weight2)  # Second matrix multiplication
        x = torch.matmul(x, self.weight3)  # Third matrix multiplication
        x = x + self.bias  # Add the bias

        return x