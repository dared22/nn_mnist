import toml
import torch
import torch.nn as nn
from typing import List
from pathlib import Path

# ----- Layers -----
from lowrank.layers.vanilla_low_rank import VanillaLowRankLayer
from lowrank.layers.dense_layer import DenseLayer
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer

# ----- Optimizers -----
from lowrank.optimizers.SGD import SimpleSGD
from lowrank.optimizers.DynamO import DynamicLowRankOptimizer


class ConfigParser:
    def __init__(self, path):
        """
        Initialize the parser with the path to the TOML configuration file.

        Parameters
        ----------
        path : str
            The file path to the TOML configuration file.

        Attributes
        ----------
        path : str
            Stores the path to the configuration file.
        config : dict
            Stores the loaded configuration data.

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
            Load and parse the configuration from the TOML file. If a value is not found, a default value is used.
            Proper error handling is implemented to manage potential file reading and parsing issues.
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
            print(f"Error creating layer {layer_type}: {e}")
            return None

    def create_multiple_layers(self):
        """
        Create a Feed Forward Network (FFN) based on the layer configuration.

        Returns
        -------
        model : torch.nn.Module
            The constructed FFN.
        """
        layers = []
        for layer_config in self.layers_config:
            layer = self.create_layer(layer_config)
            if layer:
                layers.append(layer)

        self.layers = layers
        return layers
    
    def parse_optimizer_config(self, optimizer_config_raw):
        self.optimizer_config = {}
        for layer_type, opt_config in optimizer_config_raw.items():
            optimizer_class = self.optimizer_class_mapping.get(opt_config['type'].strip().lower())
            layer_class = self.layer_class_mapping.get(layer_type.strip().lower())
            if optimizer_class:
                # Map string keys to (optimizer_class, parameters) tuples
                self.optimizer_config[layer_class] = (optimizer_class, opt_config.get('parameters', {}))

    def add_layer_mapping(self, layer_type, layer_class):
        self.layer_class_mapping[layer_type] = layer_class

    def add_optimizer_mapping(self, optimizer_type, optimizer_class):
        self.optimizer_class_mapping[optimizer_type] = optimizer_class

    def add_activation_mapping(self, activation_type, activation_class):
        self.activation_mapping[activation_type] = activation_class

if __name__ == "__main__":
    config = ConfigParser('/Users/leoquentin/Documents/Programmering/project-inf202/src/lowrank/config_utils/config_ex_ffn_low_rank.toml')
    print(config.optimizer_config)