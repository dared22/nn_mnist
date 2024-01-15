import toml
import torch
import torch.nn as nn
from typing import List
from pathlib import Path
from lowrank.layers.vanilla_low_rank import VanillaLowRankLayer
from lowrank.layers.dense_layer import DenseLayer

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
        self.learning_rate: float
        self.batch_size: int
        self.num_epochs: int
        self.architecture: str
        self.layers: list
        self.load_config()

    def load_config(self):
        """
        Load and parse the configuration from the TOML file. If value is not found, use default value.
        """
        try:
            with open(self.path, 'r') as file:
                config = toml.load(file)
                self.learning_rate = config['settings'].get('learningRate', 3e-4)
                self.batch_size = config['settings'].get('batchSize', 64)
                self.num_epochs = config['settings'].get('numEpochs', 10)
                self.architecture = config['settings'].get('architecture', 'ffn').lower().strip()
                self.layers_config = config.get('layer', [])
                self.create_multiple_layers()
        except FileNotFoundError as e:
            print(f"Error: {e}")
        
    def get_layer_params(self, layer_config, required_params):
        """
        Helper function to extract required parameters from layer_config.
        Returns None if any required parameter is missing.
        """
        params = {param: layer_config.get(param, None) for param in required_params}
        if any(value is None for value in params.values()):
            print(f"Missing parameters for {layer_config['type']} layer: {params}")
            return None
        params['activation'] = self.get_activation(layer_config.get('activation', None))
        return params

    def create_layer(self, layer_config):
        """
        Create a layer based on the layer configuration.
        """
        layer_type = layer_config.get('type', 'linear').strip().lower()

        if layer_type == 'dense':
            params = self.get_layer_params(layer_config, ["dims"])
            if params is not None:
                input_size, output_size = params['dims']
                return DenseLayer(input_size, output_size, params['activation'])
                
        elif layer_type == 'lowrank':
            params = self.get_layer_params(layer_config, ["dims", 'rank'])
            if params is not None:
                input_size, output_size = params['dims']
                return VanillaLowRankLayer(input_size, output_size, params['rank'], params['activation'])
    
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

    def get_activation(self, activation_name):
        """
        Get the activation function based on its name.

        Parameters
        ----------
        activation_name : str
            The name of the activation function.

        Returns
        -------
        activation : torch.nn.Module
            The corresponding PyTorch activation function.

        """
        activations = {
            'relu': nn.ReLU(),
            'linear': nn.Identity(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        try:
            return activations[activation_name]
        except KeyError:
            print(f"Unknown activation function: {activation_name}")

if __name__ == "__main__":
    config = ConfigParser('/Users/leoquentin/Documents/Programmering/project-inf202/src/lowrank/config_utils/config_ex_ffn_low_rank.toml')
    model = config.create_model()
    print(model)