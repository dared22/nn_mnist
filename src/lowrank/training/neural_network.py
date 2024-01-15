import torch
import torch.nn as nn
from lowrank.config_utils.config_parser import ConfigParser

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
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList(layers)

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
            X = self.flatten(X)
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
        # Use the parser to create FFN configuration
        # Extract layers from the configuration
        # Create an instance of FeedForward with these layers
        return FeedForward(config_parser.create_multiple_layers())
    

