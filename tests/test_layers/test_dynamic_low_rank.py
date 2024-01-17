import pytest
import torch
import torch.nn as nn
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer

def test_initialization_with_valif_parameters():
    layer = DynamicLowRankLayer(10, 20, 5)
    assert isinstance(layer, DynamicLowRankLayer)

def test_rank_constrait():
    with pytest.raises(ValueError)
        DynamicLowRankLayer(10, 20, 21) # Rank is larger than min(input_size, output_size)

def test_forward_pass():
    input_tensor = torch.randn(3, 10)   # Batch size 3, input size 10
    layer = DynamicLowRankLayer(10, 20, 5)
    output = layer(input_tensor)
    assert output.shape == (3, 20)      # Expected output shape with batch size 3 and output size 20

def test_bias_and_parameter_shapes():
    layer = DynamicLowRankLayer(10, 20, 5)
    assert layer.U.shape == (10, 5)
    assert layer.S.shape == (5, 5)
    assert layer.V.shape == (20, 5)
    assert layer.bias.shape == (20,)

def test_activation_function():
    layer = DynamicLowRankLayer(10, 20, 5, activation=nn.ReLU())
    input_tensor = torch.randn(3, 10)
    output = layer(input_tensor)
    assert (output >=0).all()   # Check if ReLU activation is applied 

def test_parameter_initialization():
    layer = DynamicLowRankLayer(10, 20, 5)
    assert torch.any(layer.U != 0)
    assert torch.any(layer.S != 0)
    assert torch.any(layer.V != 0)
    assert torch.any(layer.bias != 0)



