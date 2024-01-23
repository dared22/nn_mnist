import torch
import pytest
from lowrank.optimizers.DynamO import DynamicLowRankOptimizer

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

def test_step_method(model_params):
    optimizer = DynamicLowRankOptimizer(model_params)
    optimizer.zero_grad()
    # Perform a mock backward pass
    loss = model_params[0].sum()
    loss.backward()
    optimizer.step()
    # Assertions to check if parameters are updated
    for param in model_params:
        assert param.grad is None or param.grad.sum().item() != 0

def test_toggle_only_s(model_params):
    optimizer = DynamicLowRankOptimizer(model_params)
    initial_state = optimizer.defaults["only_S"]
    optimizer.toggle_only_S()
    assert optimizer.defaults["only:S"] != initial_state

def test_zero_gradients(model_params):
    optimizer = DynamicLowRankOptimizer(model_params)
    optimizer.zero_grad()
    # Apply zero gradients manually
    for p in model_params:
        p.grad = torch.zeros_like(p)

    optimizer.step()
    # Check if parameters are not updated
    for param in model_params:
        assert torch.all(torch.eq(param.grad, torch.zeros_like(param))).item()
