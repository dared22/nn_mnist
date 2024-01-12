import torch
from torchvision import datasets, transforms
import os

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download and load the MNIST training data
mnist_train = datasets.MNIST(root='./data', download=True, train=True, transform=transform)

# Download and load the MNIST test data
mnist_test = datasets.MNIST(root='./data', download=True, train=False, transform=transform)

# Save the datasets
torch.save(mnist_train, './data/mnist_train.pt')
torch.save(mnist_test, './data/mnist_test.pt')
