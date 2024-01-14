import torch
import torch.nn as nn
from lowrank.training.neural_network import FeedForward

def test_neural_network_instantation():
    # Test instantation with a simple configuration
    layers = [nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)]
    model = FeedForward(layers)
    assert isinstance(model, nn.Module)
    assert len(model.layers) == 3


def test_forward_pass():
    # Test the forward pass with a dummy input
    layers = [nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)]
    model = FeedForward(layers)
    input_tensor = torch.randn(1, 10)   # Batch size of 1
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (1, 2)

def test_layer_composition():
    # Test if layer are correctly composed in the network
    layers = [nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)]
    model = FeedForward(layers)
    assert isinstance(model.layers[0], nn.Linear)
    assert isinstance(model.layers[1], nn.ReLU)
    assert isinstance(model.layers[2], nn.Linear)

    # Check the parameters of the layers

    assert model.layers[0].in_features == 10
    assert model.layers[0].out_features == 5
    assert model.layers[2].in_features == 5
    assert model.layers[2].out_features == 2

def test_gradient_flow():
    layers = [nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)]
    model = FeedForward(layers)
    input_tensor = torch.randn(1, 10, requires_grad=True)
    output_tensor = model(input_tensor)
    output_tensor.sum().backward()
    assert input_tensor.grad is not None

def test_output_for_known_input():
    torch.manual_seed(0)     # for reproducibility
    layers = [nn.Linear(2, 2, bias=False)]
    for layer in layers:
        nn.init.eye_(layer.weight)  # Initializes with identity matrix
    model = FeedForward(layers)
    input_tensor = torch.tensor([[1., 2.]])
    expected_output = torch.tensor([[1., 2.]])
    output_tensor = model(input_tensor)
    assert torch.allclose(output_tensor, expected_output)

def test_single_layer_network():
    layer = nn.Linear(10, 5)
    model = FeedForward([layer])
    assert len(model.layers) == 1
    