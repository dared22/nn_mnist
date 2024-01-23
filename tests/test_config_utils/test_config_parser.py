import pytest
import torch.nn as nn
from lowrank.layers.vanilla_low_rank import VanillaLowRankLayer
from lowrank.layers.dense_layer import DenseLayer
from pathlib import Path
from lowrank.config_utils.config_parser import ConfigParser
import toml
from pathlib import Path
import tempfile
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.layers.dense_layer import DenseLayer
from lowrank.optimizers.dynamic_low_rank_optimizer import DynamicLowRankOptimizer


@pytest.fixture
def valid_config_path(tmp_path):
    config = {
        'settings': {'batchSize': 32, 'numEpochs': 20, 'architecture': 'ffn'},
        'layer': [{'type': 'dense', 'dims': [128, 64], 'activation': 'relu'}],
        'optimizer': {'dense': {'type': 'simplesgd', 'parameters': {'lr': 0.01}}}
    }
    config_file = tmp_path / "config.toml"
    with open(config_file, 'w') as file:
        toml.dump(config, file)
    return config_file

@pytest.fixture
def invalid_config_path():
    return 'nonexistent_path.toml'

# Test Initialization with a valid configuration file
def test_initialization_with_valid_config(valid_config_path):
    parser = ConfigParser(valid_config_path)
    assert parser.batch_size == 32
    assert parser.num_epochs == 20
    assert parser.architecture == 'ffn'
    # Add more assertions based on the expected state of the parser

# Test Initialization with an invalid configuration file
def test_initialization_with_invalid_config(invalid_config_path):
    with pytest.raises(FileNotFoundError):
        _ = ConfigParser(invalid_config_path)

#def create_layer

def test_create_layer_error_handling(valid_config_path):
    parser = ConfigParser(valid_config_path)
    invalid_layer_config = {'type': 'unknown_layer'}
    with pytest.raises(TypeError):
        _ = parser.create_layer(invalid_layer_config)

# Test Layer and Optimizer Class Mappings
def test_layer_optimizer_class_mappings(valid_config_path):
    parser = ConfigParser(valid_config_path)
    # Assuming VanillaLowRankLayer and DynamicLowRankOptimizer are valid types
    assert parser.layer_class_mapping['vanillalowrank'] == VanillaLowRankLayer
    assert parser.optimizer_class_mapping['dynamiclowrankoptimizer'] == DynamicLowRankOptimizer

# Test Activation Function Mappings
def test_activation_function_mappings(valid_config_path):
    parser = ConfigParser(valid_config_path)
    layer_config = {'type': 'dense', 'dims': [128, 64], 'activation': 'relu'}
    layer = parser.create_layer(layer_config)
    assert isinstance(layer.activation, nn.ReLU)  # Assuming the DenseLayer class uses an 'activation' attribute

# Test Configurations with Missing or Optional Fields
def test_optional_config_fields(valid_config_path):
    parser = ConfigParser(valid_config_path)
    # Assuming default values for batch size and num epochs
    assert parser.batch_size == 32  # Default specified in the fixture
    assert parser.num_epochs == 20  # Default specified in the fixture