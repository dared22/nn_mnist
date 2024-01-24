from lowrank.layers import DenseLayer
import torch

def test_initialization():
    layer  = DenseLayer(10, 5) 

    # Check if weights are initialized to right shape
    assert layer.weight.shape == (10, 5)
    assert layer.bias.shape == (5,)

def layer():
    input_size = 10
    output_size = 5
    return DenseLayer(input_size, output_size) 

def test_forward_pass():
    # Create a dummy input
    layer1 = layer()
    x = torch.randn(1, 10)
    output = layer1(x)

    # Check if output is computed
    assert output is not None

def test_output_shape():
    # Create a dummy input
    layer1 = layer()
    x = torch.randn(1, 10)
    output = layer1(x)

    # Check the shape of output
    assert output.shape == (1, 5)

def test_different_input_sizes():
    # Test with a different input size
    x = torch.randn(1, 20)
    layer = DenseLayer(20, 5)  # Adjust rank as needed
    output = layer(x)
    assert output.shape == (1, 5)

def test_gradient_flow():
    layer1 = layer()
    x = torch.randn(1, 10, requires_grad=True)
    output = layer1(x)
    output.sum().backward()
    assert x.grad is not None