import torch
import torch.nn as nn
import pytest
from lowrank.optimizers.meta_optimizer import MetaOptimizer
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.optimizers.dynamic_low_rank_optimizer import DynamicLowRankOptimizer
from lowrank.optimizers.sim

class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.conv = nn.Conv2d(1, 20, 5)
        self.dynamic_low_rank = DynamicLowRankLayer(20, 10)  # Example parameters
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = self.conv(x)
        x = self.dynamic_low_rank(x)
        x = self.fc(x)
        return x

# Test 1: Initialization with default configuration
def test_meta_optimizer_initialization_default():
    model = MockModel()
    optimizer = MetaOptimizer(model)
    assert isinstance(optimizer, MetaOptimizer), "MetaOptimizer not initialized correctly with default configuration"

# Test 2: Assignment of layer-specific optimizers
def test_layer_specific_optimizers_assignment():
    model = MockModel()
    optimizer_config = {
        DynamicLowRankLayer: (DynamicLowRankOptimizer, {'lr': 3e-4}),
        nn.Linear: (SimpleSGD, {'lr': 1e-3})
    }
    optimizer = MetaOptimizer(model, optimizer_config=optimizer_config)
    # Check if the correct optimizer is assigned for DynamicLowRankLayer and nn.Linear
    for name, layer in model.named_modules():
        if isinstance(layer, DynamicLowRankLayer):
            assert isinstance(optimizer.layer_optimizers[name], DynamicLowRankOptimizer), "DynamicLowRankLayer not assigned DynamicLowRankOptimizer"
        elif isinstance(layer, nn.Linear):
            assert isinstance(optimizer.layer_optimizers[name], SimpleSGD), "nn.Linear not assigned SimpleSGD"

# Test 3: Standard and alternating step
def test_meta_optimizer_step():
    model = MockModel()
    optimizer = MetaOptimizer(model, alternating_training=False)
    optimizer.step()
    # Add assertions to check if parameters are updated

    optimizer = MetaOptimizer(model, alternating_training=True)
    optimizer.step()
    # Add assertions to check if parameters are updated correctly in alternating mode

# Test 4: Zero gradients
def test_meta_optimizer_zero_grad():
    model = MockModel()
    optimizer = MetaOptimizer(model)
    # Perform a mock forward and backward pass
    input = torch.randn(1, 1, 28, 28)
    output = model(input)
    loss = output.sum()
    loss.backward()
    # Check if gradients are non-zero
    for param in model.parameters():
        assert param.grad is not None, "Gradients should be non-zero before calling zero_grad"
    # Call zero_grad and check if gradients are cleared
    optimizer.zero_grad()
    for param in model.parameters():
        assert param.grad is None or torch.equal(param.grad, torch.zeros_like(param.grad)), "Gradients not cleared by zero_grad"
