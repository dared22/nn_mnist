import torch
from lowrank.layers.vanilla_low_rank import VanillaLowRankLayer


def test_instatiation():
    layer = VanillaLowRankLayer(10, 5, 3)
    assert layer.weight1.shape == (10, 3)
    assert layer.weight2.shape == (3, 3)
    assert layer.weight3.shape == (3, 5)
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
    assert torch.allclose(layer.weight1.std(), std_dev, atol=1e-2)
    assert torch.allclose(layer.weight2.std(), std_dev, atol=1e-2)
    assert torch.allclose(layer.weight3.std(), std_dev, atol=1e-2)
    # Bias initalization check
    assert torch.mean(layer.bias).item() != 0 # just check id it's not all zeros

def test_gradient_flow():
    input_size, output_size, rank = 10, 5, 3
    layer = VanillaLowRankLayer(input_size, output_size, rank)
    input_tensor = torch.randn(1, input_size, requires_grad=True)
    output_tensor = layer(input_tensor)
    output_tensor.sum().backward()
    assert input_tensor.grad is not None
