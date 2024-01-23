import torch
import torch.nn as nn
import torch.nn.init as init

class DynamicLowRankLayer(nn.Module):
    """
    A dynamic low-rank layer for neural networks implemented in PyTorch.

    This layer decomposes a matrix into low-rank matrices for efficient computation. 
    Needs to be trained by a dynamic low rank optimizer for full functionality.
    The layer supports an optional non-linear activation function.

    Parameters
    ----------
    input_size : int
        The size of each input sample.
    output_size : int
        The size of each output sample.
    rank : int
        The rank for the low-rank approximation. Must be less than or equal to the 
        minimum of input_size and output_size.
    activation : callable, optional
        A PyTorch activation function (e.g., nn.ReLU()), applied to the output. 
        If None, no activation is applied. Default is None.

    Attributes
    ----------
    U : torch.nn.Parameter
        The left matrix in the low-rank approximation.
    V : torch.nn.Parameter
        The right matrix in the low-rank approximation.
    S : torch.nn.Parameter
        The square center matrix in the low-rank approximation.
    bias : torch.nn.Parameter
        The bias vector.

    Raises
    ------
    ValueError
        If the rank is larger than the minimum of input_size and output_size.

    Notes
    -----
    Must be trained by a dynamic low rank optimizer for the values to be updated correctly.
    Otherwise, this layer will behave excactly like a vanilla low rank layer.
    """
    def __init__(self, input_size, output_size, rank, activation=None):
        super(DynamicLowRankLayer, self).__init__()
        # Initialize U, S, and V
        if rank > min(input_size, output_size):
            raise ValueError("The rank cannot be larger than the minimum of input_size and output_size.")
        
        # Create a larger matrix for QR decomposition
        A = torch.randn(input_size + output_size, rank)

        # Perform QR decomposition
        Q, _ = torch.linalg.qr(A, 'reduced')

        # Split the matrix into U and V (that are orthogonal)
        self.U = nn.Parameter(Q[:input_size, :])
        self.V = nn.Parameter(Q[input_size:, :])
        self.S = nn.Parameter(torch.randn(rank, rank))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.activation = activation

    def forward(self, x):
        """
        Computes the output of the DynamicLowRankLayer.

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
        x = x + self.bias
        if self.activation:
            x = self.activation(x)
        return x
