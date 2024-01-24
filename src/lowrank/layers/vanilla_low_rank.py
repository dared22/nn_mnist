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
    weight2 : torch.nn.Parameter
        The second weight matrix of the layer, a square matrix with dimensions (rank, rank).
    weight3 : torch.nn.Parameter
        The third weight matrix with dimensions (rank, output_size). 
    bias : torch.nn.Parameter
        The bias parameter of the layer, with the same number of elements as the output_size.
    activation : callable (optional)
        The activation function to apply after the linear transformation
    """

    def __init__(self, input_size, output_size, rank, activation=None):
        super(VanillaLowRankLayer, self).__init__()
        # Initialize U, S, and V
        if rank > min(input_size, output_size):
            raise ValueError("The rank cannot be larger than the minimum of input_size and output_size.")
        
        # Create a larger matrix for QR decomposition
        A = torch.randn(input_size + output_size, rank)

        # Perform QR decomposition
        Q, _ = torch.linalg.qr(A, 'reduced')

        # Split the matrix into U and V
        self.U = nn.Parameter(Q[:input_size, :])
        self.V = nn.Parameter(Q[input_size:, :])
        self.S = nn.Parameter(torch.randn(rank, rank))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.activation = activation

    def forward(self, x):
        """
        Computes the output of the VanillaLowRankLayer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output of the layer.
        """
        x = torch.matmul(x, self.U)
        x = torch.matmul(x, self.S)
        x = torch.matmul(x, self.V.t())
        x = x + self.bias # Add the bias
        if self.activation:
            x = self.activation(x)
        return x