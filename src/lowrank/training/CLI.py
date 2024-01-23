# import toml
# import argparse
# import torch
# import torch.nn as nn
# from pathlib import Path
# from lowrank.layers.vanilla_low_rank import VanillaLowRankLayer
# from lowrank.layers.dense_layer import DenseLayer
# from lowrank.config_utils.config_parser import ConfigParser
# from lowrank.training.neural_network import FeedForward

# def parse_config(config_path):
#     with open(config_path, 'r') as f:
#         config = toml.load(f)

# def main(config_path):
#     # Set up the argument parser
#     parser = argparse.ArgumentParser(description='Load a TOML configuration file for a neureal network')
#     parser.add_argument('config_path', type=str, help='The path to the TOML configuration file')

#     # Parse the arguments
#     args = parser.parse_args()

#     #Create the ConfigParser instance
#     config_parser = ConfigParser(args.config_path)
#     # Parse the TOML configuration file
#     config = parse_config(config_path)

#     # Extractsettings and architecture from the configuration
#     settings = config.get('settings', {})
#     layers = config.get('architecture', {}).get('layers', [])

#     # Build ans compile the model with the settings and layers
#     model = FeedForward(settings, layers)

# if __name__ == "__main__":
#     main()

# import argparse

# def parse_input():
#     parser = argparse.ArgumentParser(description='This is a help message')
#     parser.add_argument('-v', '--value', default='default_value', help='Put in a value')
#     parser.add_argument('--flag', action='store_true', help='Set this if flag should be true')

#     args = parser.parse_args()
#     value = args.value
#     flag = args.flag

#     return value, flag

# if __name__ == '__main__':
#     a, flag = parse_input()

#     print("a =", a)
#     print("flag:", flag)


import argparse

parser = argparse.ArgumentParser(description='Load a .toml configurationfile for a neural network')
parser.add_argument('config_path', type=str, help='The path to the .toml configuration file')

arguments = parser.parse_args()
