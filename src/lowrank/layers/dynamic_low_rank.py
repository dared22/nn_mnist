import torch
import torch.nn as nn
import torch.nn.init as init

class DynamicLowRankLayer(nn.Module):
    def __init__(self, input_size, output_size, rank, activation=None):
        super(DynamicLowRankLayer, self).__init__()
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
        x = torch.matmul(x, self.U)
        x = torch.matmul(x, self.S)
        x = torch.matmul(x, self.V.t())
        x = x + self.bias
        if self.activation:
            x = self.activation(x)
        return x
