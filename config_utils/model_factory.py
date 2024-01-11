import torch
import torch.nn as nn
from torchsummary import summary

class ModelFactory:
    def __init__(self, config):
        """
        Initialize the ModelFactory with a configuration.

        Parameters
        ----------
        config : ConfigParser
            An instance of ConfigParser containing model configuration.

        """
        self.config = config
        
    def create_model(self):
        """
        Create a neural network model based on the configuration.

        Returns
        -------
        model : torch.nn.Module
            The constructed neural network model.

        """
        model_type = self.config['settings']['architecture'].strip().lower()

        if model_type == 'ffn':
            return self.create_ffn()
        elif model_type == 'cnn':
            # Pass for now, later implement custom model.
            pass
        else:
            print(f"Unknown model type: {model_type}")
            return None

    def create_ffn(self):
        """
        Create a neural network model based on the configuration.

        Returns
        -------
        model : torch.nn.Module
            The constructed neural network model.

        """
        layers = []
        for layer_config in self.config['layer']:
            layer = self.create_layer(layer_config)
            if layer:
                layers.append(layer)

        # Sequentially stack the layers
        model = nn.Sequential(*layers)

        return model

    def create_layer(self, layer_config):
        """
        Create a layer based on the layer configuration.

        Parameters
        ----------
        layer_config : dict
            A dictionary containing the configuration for a layer.

        Returns
        -------
        layer : torch.nn.Module
            The constructed layer based on the configuration.

        """
        layer_type = layer_config['type'].strip().lower()
        activation = self.get_activation(layer_config['activation'])

        if layer_type == 'dense':
            return nn.Sequential(
                nn.Linear(layer_config['dims'][0], layer_config['dims'][1]),
                activation
            )
        elif layer_type == 'lowrank':
            # Pass for now, later implement custom layer.
            pass
        
        elif layer_type == 'vanillalowrank':
            # Pass for now, later implement custom layer.
            pass
        
        else:
            print(f"Unknown layer type: {layer_type}")
            return None

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
        return activations.get(activation_name, nn.Identity())


# Example Usage
# Assuming you have a ConfigParser instance with the path to the configuration file
if __name__ == "__main__":
    from config_parser import ConfigParser
    path = '/Users/leoquentin/Documents/Programmering/project-inf202/config_utils/config_ex_ffn.toml'
    config_parser = ConfigParser(path)
    model_factory = ModelFactory(config_parser)
    model = model_factory.create_model()
    summary(model, (784,))

