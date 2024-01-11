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
        The size of each input sample.
    hidden_size1 : int
        The size of the first hidden layer.
    hidden_size2 : int
        The size of the second hidden layer.
    output_size : int
        The size of each output sample.

    Attributes
    ----------
    weight1 : torch.nn.Parameter
        The weight matrix between the input layer and the first hidden layer.
    weight2 : torch.nn.Parameter
        The weight matrix between the first and the second hidden layer.
    weight3 : torch.nn.Parameter
        The weight matrix between the second hidden layer and the output layer.
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

# Example usage
input_size = 784
output_size = 10
rank = 5


# Creating an instance of the custom layer
vanilla_low_rank_layer = VanillaLowRankLayer(input_size, output_size, rank)

# Example input tensor
input_tensor = torch.randn(1, input_size)

# Forward pass
output = vanilla_low_rank_layer(input_tensor)
print(output)
