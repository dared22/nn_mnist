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

def test_parameter_initalization():
    input_size, output_size, rank = 10, 5, 3
    layer = VanillaLowRankLayer(input_size, output_size, rank)
    # He initialization standard deviation
    
    std_dev = (2. / input_size)**0.5
    std_dev_tensor = torch.tensor(std_dev)

    assert torch.allclose(layer.U.std(), std_dev_tensor, atol=True)
    assert torch.allclose(layer.S.std(), std_dev_tensor, atol=True)
    assert torch.allclose(layer.V.std(), std_dev_tensor, atol=True)
    # Bias initalization check
    assert torch.mean(layer.bias).item() != 0 # just check id it's not all zeros


def test_gradient_flow():
    input_size, output_size, rank = 10, 5, 3
    layer = VanillaLowRankLayer(input_size, output_size, rank)
    input_tensor = torch.randn(1, input_size, requires_grad=True)
    output_tensor = layer(input_tensor)
    output_tensor.sum().backward()
    assert input_tensor.grad is not None
