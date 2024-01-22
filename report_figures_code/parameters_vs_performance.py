from lowrank.training.neural_network import FeedForward
from lowrank.training.trainer import Trainer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import toml

# ------------------ Utility functions ------------------

def params_in_dense(input, out):
	return input * out + out

def params_in_lowrank(input, out, rank):
	return input * rank + rank * rank + rank * out + out

def params_in_basedynamic(rank):
	layer_sizes = [(784, 256), (256, 128), (128, 64), (64, 10)]
	params = 0
	for x, y in layer_sizes[:-1]:
		params += params_in_lowrank(x, y, rank)
	params += params_in_dense(layer_sizes[-1][0], layer_sizes[-1][1])
	return params

total_params_dense = sum(params_in_dense(x,y) for x,y in [(784, 256), (256, 128), (128, 64), (64, 10)])
print(total_params_dense)

# ------------------ Data Prep ------------------

# Define the transformation pipeline
# Include scaling, random rotation, and random translation
# Transformation for training data (with data augmentation)
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
])

# Transformation for test data (without random transformations)
test_transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the MNIST dataset with the defined transformations
train_data = datasets.MNIST(root='data', train=True, transform=train_transform, download=True)
test_data = datasets.MNIST(root='data', train=False, transform=test_transform, download=True)

# Create training and test dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)  # Usually, shuffling is not needed for test data


print("Current Working Directory: ", os.getcwd())
# ------------------ Creating config files------------------

def modify_and_save_config(original_config_path, new_config_dir, rank):
    # Load the original configuration file
    with open(original_config_path, 'r') as file:
        config = toml.load(file)

    # Update the rank for each lowRank layer
    for layer in config.get('layer', []):
        if layer['type'] == 'lowRank':
            layer['rank'] = rank

    # Create a new config file name based on the rank
    new_config_filename = f"config_rank_{rank}.toml"
    new_config_path = os.path.join(new_config_dir, new_config_filename)

    # Save the modified configuration to a new file
    with open(new_config_path, 'w') as file:
        toml.dump(config, file)

# Specify the original config file path and the directory to save new configs
original_config_path = 'report_figures_code/config_files/basedynamiclowrank.toml'  # Replace with your config file path
new_config_dir = 'report_figures_code/config_files'  # Directory to store new config files
os.makedirs(new_config_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Create a new config file for each rank from 1 to 64
for rank in range(5, 50):
    modify_and_save_config(original_config_path, new_config_dir, rank)


def train_model(config_path):
    # Create model and trainer from config
    model = FeedForward.create_from_config(config_path)
    trainer = Trainer(model)

    # Train the model for one epoch and return validation accuracy
    # Assuming train function returns validation accuracy
    val_accuracy = trainer.train(train_loader, test_loader)
    return val_accuracy

# ------------------ Training ------------------
config_path = "report_figures_code/config_files/config_rank_49.toml"
val_accuracy = train_model(config_path)
print(f"Rank: {rank}, Validation Accuracy: {}")
