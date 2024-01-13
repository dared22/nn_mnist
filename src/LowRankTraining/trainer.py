import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from MNIST_downloader import Downloader

class Trainer:
    """
    A class for training neural network models using the MNIST dataset.

    This class facilitates the training and evaluation of neural network models on the MNIST dataset.
    It handles data loading, training loop, and validation of the model.

    Attributes:
        batchSize (int): The number of samples per batch to load.
        trainloader (DataLoader): DataLoader for the training dataset.
        testloader (DataLoader): DataLoader for the testing dataset.

    Methods:
        train: Trains a neural network model on the MNIST dataset.
    """

    def __init__(self, batch_size):
        """
        Initializes the Trainer with a specified batch size and sets up data loaders for the MNIST dataset.

        Args:
            batch_size (int): The number of samples per batch to load.
        """
        self.batchSize = batch_size

        downloader = Downloader()
        traindataset, testdataset = downloader.get_data()
        self.trainloader = DataLoader(traindataset, batch_size=self.batchSize, shuffle=True)
        self.testloader = DataLoader(testdataset, batch_size=self.batchSize, shuffle=False)
    
    def train(self, numIterations, lr, NeuralNet):
        """
        Trains a neural network model using the specified parameters and data loaders.

        This method runs the training loop for a given number of iterations, updates model parameters,
        and evaluates the model's performance on the test dataset.

        Args:
            numIterations (int): Number of iterations to train the model.
            lr (float): Learning rate for the optimizer.
            NeuralNet (nn.Module): The neural network model to be trained.

        Returns:
            nn.Module: The trained neural network model.
        """
        optimizer = torch.optim.Adam(NeuralNet.parameters(), lr=lr) # Optimizer
        
        for i in range(numIterations):
            for step, (images, labels) in enumerate(self.trainloader):
                optimizer.zero_grad()  # Reset gradients to zero
                out = NeuralNet(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
        
                if (step + 1) % 100 == 0:
                    print(f'Epoch [{i+1}/{numIterations}], Step [{step+1}/{len(self.trainloader)}], Loss: {loss.item():.4f}')
        
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.testloader:
                    outputs = NeuralNet(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
            accuracy = correct / total
            print(f'Epoch [{i+1}/{numIterations}], Validation Accuracy: {100 * accuracy:.2f}%')

        return NeuralNet
     
