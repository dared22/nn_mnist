import torch
import torch.nn as nn
from lowrank.config_utils.config_parser import ConfigParser
import os
import glob

class FeedForward(nn.Module):
    """
    A feedforward neural network module.

    This class represents a simple feedforward neural network. It is a subclass of nn.Module and 
    contains a sequence of layers through which the input data is propagated.

    Parameters
    ----------
    layers : list
        A list of neural network layers (e.g., Linear, ReLU) that will be applied in sequence.

    Attributes
    ----------
    layers : nn.ModuleList
        A ModuleList containing the layers of the neural network.

    Methods
    -------
    forward(X)
        Propagates the input through each layer of the network.
    
    create_from_config(path)
        Static method to create a FeedForward instance from a configuration file.
    export_model(path) 
        Exports a trained neural network model's state dictionary to the specified path.
    """

    def __init__(self, layers):
        """
        Initialize the FeedForward module.

        Parameters
        ----------
        layers : list
            A list of neural network layers to be applied in sequence.
        """
        super(FeedForward, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.config_parser = None

    def forward(self, X):
        """
        Forward pass through the neural network.

        This method sequentially applies each layer to the input data.

        Parameters
        ----------
        X : tensor
            The input tensor to be fed through the network.

        Returns
        -------
        tensor
            The output tensor after being processed by the network.
        """

        for layer in self.layers:
            X = layer(X)
        return X


    @staticmethod
    def create_from_config(path):
        """
        Create a FeedForward instance from a configuration file.

        This static method uses a configuration parser to interpret the configuration file specified by `path`.
        It creates and returns a FeedForward instance based on the layers defined in the configuration.

        Parameters
        ----------
        path : str
            The path to the configuration file.

        Returns
        -------
        FeedForward
            An instance of the FeedForward class configured as per the configuration file.
        """
        # Instantiate your configuration parser
        config_parser = ConfigParser(path)
        # Extract layers from the configuration
        # Create an instance of FeedForward with these layers
        model = FeedForward(config_parser.layers)
        model.config_parser = config_parser
        return model

    def export_model(self, trained_nn, path):
        """
        Exports a trained neural network model's state dictionary to the specified path.

        Args:
            trained_nn (torch.nn.Module): The trained neural network model whose state dictionary is to be saved.
        """
        torch.save(trained_nn.state_dict(), path)

    def import_model(self,nn, path): 
        """
        Imports a neural network model's state dictionary from the specified path.

        Args:
            nn (class): The neural network class to be instantiated and loaded.

        Returns:
            The loaded neural network model with its state dictionary imported from the file.
        """ 
        nn.load_state_dict(torch.load(path))

    @staticmethod
    def mass_create_models(directory):
        """
        Creates a dictionary of FeedForward models from all .toml configuration files in the specified directory.

        Parameters
        ----------
        directory : str
            The path to the directory containing .toml configuration files.

        Returns
        -------
        dict
            A dictionary with filenames as keys (without the .toml extension) and instantiated FeedForward objects as values.
        """
        models = {}
        # List all .toml files in the directory
        for filepath in glob.glob(os.path.join(directory, '*.toml')):
            filename = os.path.basename(filepath).rsplit('.', 1)[0]
            model = FeedForward.create_from_config(filepath)
            models[filename] = model
        return models

  
    

