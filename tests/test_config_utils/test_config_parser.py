import pytest
import torch.nn as nn
from lowrank.layers.vanilla_low_rank import VanillaLowRankLayer
from lowrank.layers.dense_layer import DenseLayer
from pathlib import Path
from lowrank.config_utils.config_parser import ConfigParser

def test_parser():
	config_parser = ConfigParser("tests/data/config_ex_ffn.toml")

	config_parser.load_config()

	assert config_parser.config is not None

# Mock TOML files
@pytest.fixture
def valid_config_file(tmp_path):
	config = """
	[settings]
	learningrate = 0.001
	batchSize = 32
	numEpochs = 20
	architecture = 'ffn'

	[[layer]]
	type = 'dense'
	dims = [10, 20]
	activation = 'relu'

	[[layer]]
	type = 'lowrank'
	dims = [20, 30]
	rank = 5
	activation = 'tanh'
	"""
	config_path = tmp_path / "valid_config.toml"
	config_path.write_text(config)
	return config_path

@pytest.ficture
def incomplete_config_file(tmp_path):
	config = """
	[settings]
	learningrate = 0.001
	# batchSize and numEpochs missing
	
	[[layer]]
	type = 'dense'
	# dims missing
	activation = 'relu'
	"""
	config_path = tmp_path / "incomplete_config.toml"
	config_path.write_text(config)
	return config_path

def test_successful_config_loading(valid_config_file):
	parser = ConfigParser(valid_config_file)
	assert parser.learning_rate == 0.001
	assert parser.batch_size == 32
	assert parser.num_epochs == 20
	assert parser.architecture == 'ffn'
	assert isinstance(parser.layers[0], DenseLayer)
	assert isinstance(parser.layers[1], VanillaLowRankLayer)

def test_default_values(incomplete_config_file):
	parser = ConfigParser(incomplete_config_file)
	assert parser.learning_rate == 0.001
	assert parser.batch_size == 64	# Default value
	assert parser.num_epochs == 10	# Default value
	assert len(parser.layers) == 0	# No valid layers due to missing dims

def test_layer_creation(valid_config_file):
	parser = ConfigParser(valid_config_file)
	assert len(parser.layers) == 2
	assert isinstance(parser.layers[0], DenseLayer)
	assert isinstance(parser.layers[1], VanillaLowRankLayer) 

def test_incorrect_layer_type(incomplete_config_file):
	config_path = incomplete_config_file
	# Modify the file to include incorrect layer type
	config_path.write_text(config_path.read_text().replace("type = 'dense", "type = 'unknown'"))
	parser = ConfigParser(config_path)
	assert len(parser.layers) == 0	# No layers should be createed

def test_unkown_activation_function(incomplete_config_file):
	config_path = incomplete_config_file
	# Modify the file to include an unknown activation function
	config_path.write_text(config_path.read_tect().replace("activation = 'relu'", "activation = 'unknown'"))
	parser = ConfigParser(config_path)
	# Assuminf the layer is still created but with a default activation function
	assert len(parser.layers) == 0 	# No layers should be created
