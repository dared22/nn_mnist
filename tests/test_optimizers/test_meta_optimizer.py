import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.optimizers import DynamicLowRankOptimizer, SimpleSGD, MetaOptimizer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 50)
        self.dlr = DynamicLowRankLayer(50, 10, 5)

    def forward(self, x):
        x = self.fc(x)
        x = self.dlr(x)
        return x

@pytest.fixture
def simple_model():
    return SimpleModel()

@pytest.fixture
def optimizer_config():
    return {
        nn.Linear: (SGD, {'lr': 0.01}),
        DynamicLowRankLayer: (DynamicLowRankOptimizer, {'lr': 0.001})
    }

def test_meta_optimizer_initialization(simple_model, optimizer_config):
    meta_optimizer = MetaOptimizer(simple_model, optimizer_config)
    assert isinstance(meta_optimizer, MetaOptimizer)
    assert len(meta_optimizer.layer_optimizers) == 2

def test_default_optimizer_assignment(simple_model):
    meta_optimizer = MetaOptimizer(simple_model)
    assert isinstance(meta_optimizer.default_optimizer, SimpleSGD)

def test_alternating_step(simple_model, optimizer_config):
    meta_optimizer = MetaOptimizer(simple_model, optimizer_config, alternating_training=True)
    meta_optimizer.zero_grad()
    output = simple_model(torch.randn(1, 784))
    output.mean().backward()
    meta_optimizer.step()
    assert meta_optimizer.train_only_S == True  # Assuming the initial state is False

def test_standard_step(simple_model, optimizer_config):
    meta_optimizer = MetaOptimizer(simple_model, optimizer_config, alternating_training=False)
    
    # Store the initial state of the parameters
    initial_params = [param.clone() for param in simple_model.parameters()]

    # Perform a backward pass and optimization step
    meta_optimizer.zero_grad()
    output = simple_model(torch.randn(1, 784))
    output.mean().backward()
    meta_optimizer.step()

    # Check if the parameters have been updated
    for initial_param, updated_param in zip(initial_params, simple_model.parameters()):
        assert not torch.equal(initial_param, updated_param)

def test_zero_grad(simple_model, optimizer_config):
    meta_optimizer = MetaOptimizer(simple_model, optimizer_config)
    output = simple_model(torch.randn(1, 784))
    output.mean().backward()
    meta_optimizer.zero_grad()
    for param in simple_model.parameters():
        assert param.grad is None
