import torch
from lowrank.layers import DynamicLowRankLayer
from lowrank.optimizers import DynamicLowRankOptimizer
import pytest

def test_layer_initialization():
    input_size = 10
    output_size = 5
    rank = 3

    # Valid initialization
    layer = DynamicLowRankLayer(input_size, output_size, rank)
    assert layer.U.shape == (input_size, rank)
    assert layer.V.shape == (output_size, rank)
    assert layer.S.shape == (rank, rank)

    # Invalid initialization
    with pytest.raises(ValueError):
        DynamicLowRankLayer(input_size, output_size, min(input_size, output_size) + 1)

def test_forward_pass():
    input_size = 10
    output_size = 5
    rank = 3
    layer = DynamicLowRankLayer(input_size, output_size, rank)

    # Without activation
    x = torch.randn(1, input_size)
    y = layer(x)
    assert y.shape == (1, output_size)

    # With activation
    layer = DynamicLowRankLayer(input_size, output_size, rank, activation=torch.nn.ReLU())
    y = layer(x)
    assert y.shape == (1, output_size)
    
def test_optimizer_integration():
    input_size = 10
    output_size = 5
    rank = 3
    layer = DynamicLowRankLayer(input_size, output_size, rank)
    optimizer = DynamicLowRankOptimizer(layer.parameters())

    # Perform an optimization step
    x = torch.randn(1, input_size)
    y = layer(x)
    y.sum().backward()
    optimizer.step()

    # Check if parameters have gradients and have been updated
    for param in layer.parameters():
        assert param.grad is not None
        assert param.requires_grad
        
def test_optimizer_initialization_and_step():
    input_size = 10
    output_size = 5
    rank = 3
    layer = DynamicLowRankLayer(input_size, output_size, rank)
    optimizer = DynamicLowRankOptimizer(layer.parameters())
    
	# Perform an optimization step
    x = torch.randn(1, input_size)
    y = layer(x)
    y.sum().backward()

    # Test step with updating all parameters
    optimizer.step(only_S=False)

    # Test step with updating only S matrix
    optimizer.step(only_S=True)

    # Test toggling of only_S
    assert optimizer.defaults["only_S"]
    optimizer.toggle_only_S()
    assert not optimizer.defaults["only_S"]

    # Test for ValueError with invalid parameters
    with pytest.raises(ValueError):
        DynamicLowRankOptimizer(layer.parameters(), lr=-1)
        
