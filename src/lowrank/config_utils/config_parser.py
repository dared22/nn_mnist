import toml
import torch
import torch.nn as nn
from typing import List
from pathlib import Path

# ----- Layers -----
from lowrank.layers import VanillaLowRankLayer
from lowrank.layers import DenseLayer
from lowrank.layers import DynamicLowRankLayer

# ----- Optimizers -----
from lowrank.optimizers.simple_sgd import SimpleSGD
from lowrank.optimizers.dynamic_low_rank_optimizer import DynamicLowRankOptimizer


class ConfigParser:
    def __init__(self, path):
        """
        Parser for loading and handling configuration from a TOML file.

        Attributes
        ----------
        path : Path
            The path to the TOML configuration file.
        batch_size : int
            The batch size for training, read from config.
        num_epochs : int
            The number of epochs for training, read from config.
        architecture : str
            The network architecture type, read from config.
        layers : list
            The list of layer objects created based on the config.
        optimizer_config : dict
            Configuration dictionary for optimizers.

        Methods
        -------
        load_config():
            Loads and parses configuration from the TOML file.
        create_layer(layer_config):
            Creates a layer based on the configuration.
        create_multiple_layers():
            Creates multiple layers and forms a network based on the configuration.
        parse_optimizer_config(optimizer_config_raw):
            Parses and sets up optimizer configurations.
        add_layer_mapping(layer_type, layer_class):
            Adds a mapping from layer type to layer class.
        add_optimizer_mapping(optimizer_type, optimizer_class):
            Adds a mapping from optimizer type to optimizer class.
        add_activation_mapping(activation_type, activation_class):
            Adds a mapping from activation type to activation class.
        """
        self.path: Path = path
        self.batch_size: int
        self.num_epochs: int
        self.architecture: str
        self.layers: list
        self.optimizer_config: dict

        # Map from layer names (strings) to layer classes
        self.layer_class_mapping = {
            'default': 'default',
            'vanillalowrank': VanillaLowRankLayer,
            "lowrank": DynamicLowRankLayer,
            'dense': DenseLayer,
            # Add other mappings as needed
        }

        # Map from optimizer names (strings) to optimizer classes
        self.optimizer_class_mapping = {
            'simplesgd': SimpleSGD,
            'dynamiclowrankoptimizer': DynamicLowRankOptimizer,
            'adam': torch.optim.Adam,
            # Add other mappings as needed
        }

        # Map activation function names to PyTorch activation functions
        self.activation_mapping = {
            'relu': nn.ReLU(),
            'linear': nn.Identity(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }

        self.load_config()


    def load_config(self):
            """
            Load and parse the configuration from the TOML file. Handles errors related
            to file access and content parsing.
            """
            try:
                with open(self.path, 'r') as file:
                    config = toml.load(file)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Could not find configuration file at {self.path}.") from e
            except toml.TomlDecodeError as e:
                raise ValueError(f"Error parsing TOML file at {self.path}. Please ensure it is formatted correctly.") from e

            try:
                self.batch_size = config['settings'].get('batchSize', 64)
                self.num_epochs = config['settings'].get('numEpochs', 10)
                self.architecture = config['settings'].get('architecture', 'ffn').lower().strip()
                self.layers_config = config.get('layer', [])
                optimizer_config = config.get('optimizer', {})
            except KeyError as e:
                raise KeyError(f"Required configuration key missing: {e}. Please check the configuration file.")  
            
            self.create_multiple_layers()
            self.parse_optimizer_config(optimizer_config)


    def create_layer(self, layer_config):
        """
        Create a layer based on the layer configuration.
        """
        layer_type = layer_config.pop('type', 'default').strip().lower()
        layer_class = self.layer_class_mapping.get(layer_type)

        # Convert string defining activation function to actual activation function
        if 'activation' in layer_config:
            activation_name = layer_config['activation'].strip().lower()
            layer_config['activation'] = self.activation_mapping.get(activation_name)

        # Convert dims to input_size and output_size
        if 'dims' in layer_config:
            layer_config['input_size'] = layer_config['dims'][0]
            layer_config['output_size'] = layer_config['dims'][1]
            del layer_config['dims']

        # Create layer based on layer type
        try:
            return layer_class(**layer_config)
        except TypeError as e:
            raise TypeError(f"Error creating layer {layer_type}: {e}") from e
            print(f"Error creating layer {layer_type}: {e}")
            return None

    def create_multiple_layers(self):
        """
        Create a list of layers based on the configuration. The layers are created in the order they appear in the
        configuration file. The layers are stored in the `layers` attribute and returned.

        Returns
        -------
        layers : list
            A list of layer objects created based on the configuration.
        """
        layers = []
        for layer_config in self.layers_config:
            layer = self.create_layer(layer_config)
            if layer:
                layers.append(layer)

        self.layers = layers
        return layers
    
    def parse_optimizer_config(self, optimizer_config_raw):
        """
        Parses and sets up optimizer configuration dictionary for the MetaOptimizer.
        """
        self.optimizer_config = {}
        for layer_type, opt_config in optimizer_config_raw.items():
            optimizer_class = self.optimizer_class_mapping.get(opt_config['type'].strip().lower())
            layer_class = self.layer_class_mapping.get(layer_type.strip().lower())
            if optimizer_class:
                # Map string keys to (optimizer_class, parameters) tuples
                self.optimizer_config[layer_class] = (optimizer_class, opt_config.get('parameters', {}))

    def add_layer_mapping(self, layer_type, layer_class):
        """
        Add a mapping from layer type to layer class.
        """
        self.layer_class_mapping[layer_type] = layer_class

    def add_optimizer_mapping(self, optimizer_type, optimizer_class):
        """
        Add a mapping from optimizer type to optimizer class.
        """
        self.optimizer_class_mapping[optimizer_type] = optimizer_class

    def add_activation_mapping(self, activation_type, activation_class):
        """
        Add a mapping from activation type to activation class.
        """
        self.activation_mapping[activation_type] = activation_class