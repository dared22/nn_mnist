import torch
import pytest
from lowrank.optimizers import DynamicLowRankOptimizer

@pytest.fixture
def model_params():
    U = torch.randn(10, 5, requires_grad=True)
    S = torch.randn(5, 5, requires_grad=True)
    V = torch.randn(5, 10, requires_grad=True)
    bias = torch.randn(10, requires_grad=True)
    return [U, S, V, bias] 

def test_optimizer_initialization(model_params):
    optimizer = DynamicLowRankOptimizer(model_params)
    assert isinstance(optimizer, DynamicLowRankOptimizer)

    with pytest.raises(ValueError):
        DynamicLowRankOptimizer(model_params, lr=-1)

def test_invalid_parameters():
    with pytest.raises(TypeError):
        DynamicLowRankOptimizer("invalid parameters")

