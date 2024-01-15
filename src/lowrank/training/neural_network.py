import torch
import torch.nn as nn
from lowrank.config_utils.config_parser import ConfigParser

class FeedForward(nn.Module):
    def __init__(self, layers):
        super(FeedForward, self).__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X


    @staticmethod
    def create_from_config(path):
        # Instantiate your configuration parser
        config_parser = ConfigParser(path)
        # Use the parser to create FFN configuration
        # Extract layers from the configuration
        # Create an instance of FeedForward with these layers
        return FeedForward(config_parser.create_multiple_layers())


