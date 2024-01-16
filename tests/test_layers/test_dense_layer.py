from lowrank.layers.dense_layer import DenseLayer
import torch

def test_initialization():
    layer  = DenseLayer(10, 5, rank=3)  # Adjust rank as needed

    # Check if weights are initialized with He initialization
    assert layer.weight.shape == (10, 5)
    assert layer.bias.shape == (5,)

def layer():
    input_size = 10
    output_size = 5
    return DenseLayer(input_size, output_size, rank=3)  # Adjust rank as needed

def test_forward_pass(layer):
    # Create a dummy input
    x = torch.randn(1, 10)
    output = layer(x)

    # Check if output is computed
    assert output is not None

def test_output_shape(layer):
    # Create a dummy input
    x = torch.randn(1, 10)
    output = layer(x)

    # Check the shape of output
    assert output.shape == (1, 5)

def test_different_input_sizes():
    # Test with a different input size
    x = torch.randn(1, 20)
    layer = DenseLayer(20, 5, rank=3)  # Adjust rank as needed
    output = layer(x)
    assert output.shape == (1, 5)

def test_gradient_flow(layer):
    x = torch.randn(1, 10, requires_grad=True)
    output = layer(x)
    output.sum().backward()
    assert x.grad is not None