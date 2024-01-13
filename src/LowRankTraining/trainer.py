import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from MNIST_downloader import Downloader

class Trainer:
    """
    A class for training and evaluating neural network models on the MNIST dataset.

    This class manages the training process of a neural network using the MNIST dataset. It includes
    functionality for both training and validation phases and logs the training process using TensorBoard.

    Attributes:
        batchSize (int): The number of samples per batch to be loaded in the DataLoader.
        trainloader (DataLoader): DataLoader for the MNIST training dataset.
        testloader (DataLoader): DataLoader for the MNIST testing dataset.
        writer (SummaryWriter): TensorBoard SummaryWriter for logging training and validation metrics.

    Methods:
        train: Trains a neural network model and evaluates its performance on the test dataset.
    """

    def __init__(self, batch_size):
        """
        Initializes the Trainer class with the specified batch size and sets up data loaders for the MNIST dataset.

        The function also initializes a TensorBoard SummaryWriter to log training and validation metrics.

        Args:
            batch_size (int): The number of samples per batch to be loaded in the DataLoader.
        """
        self.batchSize = batch_size
        downloader = Downloader()
        traindataset, testdataset = downloader.get_data()
        self.trainloader = DataLoader(traindataset, batch_size=self.batchSize, shuffle=True)
        self.testloader = DataLoader(testdataset, batch_size=self.batchSize, shuffle=False)
        self.writer = SummaryWriter('./runs')  # TensorBoard SummaryWriter

    def train(self, numIterations, lr, NeuralNet):
        """
        Trains and evaluates a neural network model using the specified parameters.

        This method executes the training loop for a given number of iterations, performs backpropagation,
        updates model parameters, and evaluates the model on the test dataset after each epoch. Training and
        validation metrics are logged to TensorBoard.

        Args:
            numIterations (int): Number of epochs to train the model.
            lr (float): Learning rate for the optimizer.
            NeuralNet (nn.Module): The neural network model to be trained and evaluated.

        Returns:
            nn.Module: The trained neural network model.
        """
        optimizer = torch.optim.Adam(NeuralNet.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for i in range(numIterations):
            NeuralNet.train()  # Set the model to training mode
            for step, (images, labels) in enumerate(self.trainloader):
                optimizer.zero_grad()
                out = NeuralNet(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
        
                if (step + 1) % 100 == 0:
                    print(f'Epoch [{i+1}/{numIterations}], Step [{step+1}/{len(self.trainloader)}], Loss: {loss.item():.4f}')
                    self.writer.add_scalar('Training Loss', loss.item(), i*len(self.trainloader) + step)

            NeuralNet.eval()  # Set the model to evaluation mode
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
            self.writer.add_scalar('Validation Accuracy', accuracy, i+1)

        self.writer.close()  # Close the TensorBoard writer
        return NeuralNet
