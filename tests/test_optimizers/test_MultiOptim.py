import torch
import torch.nn as nn
import pytest
from lowrank.optimizers import MetaOptimizer, SimpleSGD, DynamicLowRankOptimizer
from lowrank.layers import DynamicLowRankLayer

@pytest.fixture
def simple_model():
    model = nn.Sequential(
        nn.Linear(10, 20), 
        DynamicLowRankLayer(20, 30), 
        nn.Linear(30, 40)
    )
    return model

@pytest.fixture
def optimizer_config():
    return {
        "default" : (SimpleSGD, {'lr:  3e-4'}),
        DynamicLowRankLayer: (DynamicLowRankOptimizer, {'lr:3e-4'}),
    }

def test_meta_optimizer_initializatoin(simple_model, optimizer_config):
    meta_optimizer = MetaOptimizer(simple_model, optimizer_config)
    assert isinstance(meta_optimizer, optimizer_config)
    assert len(meta_optimizer.layer_optimizers) == 1 # Assuming one dynamic layer
    assert isinstance(meta_optimizer.default_optimizer, SimpleSGD)

def test_step_function(simple_model, optimizer_config):
    meta_optimizer = MetaOptimizer(simple_model, optimizer_config)
    meta_optimizer.zero_grad()

    input_tensor = torch.randn(1, 10)
    output = simple_model(input_tensor)
    loss = output.sum()
    loss.backward()
    
    meta_optimizer.step()

    for param in simple_model.parameters():
        assert param.grad is None or param.grad.sum().item() !=0
    
def test_zero_grad_function(simple_model, optimizer_config):
    meta_optimizer = MetaOptimizer(simple_model, optimizer_config)
    input_tensor = torch.randn(1, 10)
    output = simple_model(input_tensor)
    loss = output.sum()
    loss.backward()

    meta_optimizer.zero_grad()

    for param in simple_model.parameters():
        assert param.grad is None or torch.all(torch.eq(param.grad, torch.zeros_like(param))).item()

    
    def test_individual_layer_optimizers(simple_model, optimizer_config):
        meta_optimizer = MetaOptimizer(simple_model, optimizer_config)
        for name, optimizer in meta_optimizer.layer_optimizers.items():
            assert isinstance(optimizer, DynamicLowRankOptimizer)

def test_layers_without_parameters(simple_model, optimizer_config):
    simple_model.add_module('relu', nn.ReLU())  # Adding a layer without parameters
    meta_optimizer = MetaOptimizer(simple_model, optimizer_config)
    assert 'relu' not in meta_optimizer.layer_optimizers # ReLU should not be in layer_optimizers
