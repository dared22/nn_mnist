import pytest
from lowrank.config_utils.config_parser import ConfigParser
import toml
from pathlib import Path
import tempfile
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.layers.dense_layer import DenseLayer
from lowrank.optimizers.DynamO import DynamicLowRankOptimizer


# Fixture for creating a temporary TOML file
@pytest.fixture
def temp_toml_file():
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.toml', delete=False) as f:
        content = """
[settings]
batchSize = 64
numEpochs = 10
architecture = 'FFN'

[[layer]]
type = 'lowRank'
dims = [784,512]
activation = 'relu'
rank = 20

[[layer]]
type = 'lowRank'
dims = [512, 256]
activation = 'relu'
rank = 20

[[layer]]
type = 'dense'
dims=[256,10]
activation = 'linear'

[optimizer.default]
type = 'SimpleSGD'
parameters = { lr = 0.005 }

[optimizer.lowRank]
type = 'DynamicLowRankOptimizer'
parameters = { lr = 0.1 }

[optimizer.vanillaLowRank]
type = 'SimpleSGD'
parameters = { lr = 0.005 }
"""
        f.write(content)
        f.seek(0)
        yield Path(f.name)

def test_config_loading(temp_toml_file):
    parser = ConfigParser(temp_toml_file)
    assert parser.batch_size == 64
    assert parser.num_epochs == 10
    assert parser.architecture == 'ffn'

def test_layer_creation(temp_toml_file):
    parser = ConfigParser(temp_toml_file)
    assert len(parser.layers) == 3
    assert isinstance(parser.layers[0], DynamicLowRankLayer)
    assert isinstance(parser.layers[2], DenseLayer)

def test_optimizer_config_parsing(temp_toml_file):
    parser = ConfigParser(temp_toml_file)
    parser.load_config()
    assert 'default' in parser.optimizer_config
    assert DynamicLowRankLayer in parser.optimizer_config

def test_error_handling_invalid_file():
    with pytest.raises(FileNotFoundError):
        ConfigParser('invalid_path.toml')

# Add more tests as needed
