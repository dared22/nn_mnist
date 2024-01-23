"""
Example Usage of Configuration System for Generating Feedforward Neural Network
This script demonstrates the independent use of a configuration system to create a feedforward neural network.
It operates independently from other classes like the trainer class or metaoptimizer class.
The model is defined and instantiated using a configuration file, providing a modular and easily adaptable setup.
The example focuses on using the MNIST dataset, showcasing the network's training and evaluation on this benchmark.
Note: The configuration file path and other specifics should be adjusted according to your project's structure.
"""

from lowrank.training import FeedForward
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the model from configuration
model = FeedForward.create_from_config("example_usage/dense_model_from_config/simple_dense_config.toml")

# MNIST specific transformations (0-1 scaling)
transform = transforms.Compose([
    transforms.ToTensor()  # Scales images to [0, 1] range
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./example_usage', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./example_usage', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()  # Set model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    # Validation loop
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%")

# Optionally, add more comprehensive evaluation here if needed
