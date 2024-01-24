import torch
from lowrank.layers import VanillaLowRankLayer


def test_instatiation():
    layer = VanillaLowRankLayer(10, 5, 3)
    assert layer.U.shape == (10, 3)
    assert layer.S.shape == (3, 3)
    assert layer.V.shape == (5, 3)
    assert layer.bias.shape == (5,)

def test_forward_pass():
    input_size, output_size, rank = 10, 5, 3
    layer = VanillaLowRankLayer(input_size, output_size, rank)
    input_tensor = torch.randn(1, input_size) # Batch size of 1
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (1, output_size)


def test_gradient_flow():
    input_size, output_size, rank = 10, 5, 3
    layer = VanillaLowRankLayer(input_size, output_size, rank)
    input_tensor = torch.randn(1, input_size, requires_grad=True)
    output_tensor = layer(input_tensor)
    output_tensor.sum().backward()
    assert input_tensor.grad is not None
