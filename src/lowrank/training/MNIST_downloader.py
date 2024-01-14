import torch
from torchvision import datasets, transforms
import os

class Downloader:
    def __init__(self) -> None:
        # Define a transform to normalize the data
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Download and load the MNIST training data
        self.mnist_train = datasets.MNIST(root='./data', download=True, train=True, transform=transform)

        # Download and load the MNIST test data
        self.mnist_test = datasets.MNIST(root='./data', download=True, train=False, transform=transform)

        # Save the datasets
        torch.save(self.mnist_train, './data/mnist_train.pt')
        torch.save(self.mnist_test, './data/mnist_test.pt')

    def get_data(self):
        train = self.mnist_train
        test = self.mnist_test
        return train, test



        
