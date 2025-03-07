import pytest
import torch
from lowrank.optimizers import SimpleSGD

@pytest.fixture
def mock_parameters():
    return [torch.tensor([1.0, 2.0, 3.0], requires_grad=True)]

def test_initialization_with_valid_lr(mock_parameters):
    optimizer = SimpleSGD(mock_parameters, lr=1e-3)
    assert isinstance(optimizer, SimpleSGD)

    with pytest.raises(ValueError):
        SimpleSGD(mock_parameters, lr=-1)


def test_learning_rate_impact(mock_parameters):
    lr = 1e-3
    optimizer = SimpleSGD(mock_parameters, lr=lr)
    optimizer.zero_grad()

    loss = mock_parameters[0].sum()
    loss.backward()
    
    old_values = [p.clone() for p in mock_parameters] 
    optimizer.step()

    for old_val, param in zip(old_values, mock_parameters):
        assert torch.all(torch.eq(old_val - lr * param.grad, param)).item()

def test_handling_none_gradients():
    param_with_none_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    optimizer = SimpleSGD([param_with_none_grad], lr=1e-3)
    optimizer.step()    # No backward pass, sp grad should be None

    assert torch.all(torch.eq(param_with_none_grad, torch.tensor([1.0, 2.0, 3.0]))).item()

