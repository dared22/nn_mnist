import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DenseLayer(nn.Module):
    """
    A custom layer in PyTorch representing a vanilla low rank dense layer.
    
    This layer performs matrix multiplication in succession with three different weight matrices.

    Parameters
    ----------
    input_size : int
        The size of the input.
    output_size : int
        The size of the output.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The weight matrix between the input layer and the first hidden layer. Initialized using He initialization suitable for ReLU activations.
    bias : torch.nn.Parameter
		The bias parameter of the layer, with the same number of elements as the 
		output_size. Initialized with random values from a standard normal distribution.
    """

    def __init__(self, input_size, output_size, rank):
        super(DenseLayer, self).__init__()
        # Define the weight matrix
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))
        
        # Initialize the weight using He initialization
        init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        
        # Initialize the bias with random values
        self.bias = nn.Parameter(torch.randn(output_size)) 
   
    def forward(self, x):
        """
        Defines the forward pass of the DenseLayer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after applying matrix multiplications with the weight matrices.
        """
        x = torch.matmul(x, self.weight)  
        x = x + self.bias  # Add the bias

        return x