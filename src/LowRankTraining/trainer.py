import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from MNIST_downloader import Downloader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        

    def train(self, numIterations, lr, NeuralNet, patience=2):
        """
        Trains a neural network model using the MNIST dataset.

        This method handles the training and validation process of a neural network. It employs 
        a learning rate scheduler to adjust the learning rate during training, implements early 
        stopping based on validation accuracy, and saves checkpoints of the model.

        Args:
            numIterations (int): The number of epochs to train the model.
            lr (float): The initial learning rate for the optimizer.
            NeuralNet (torch.nn.Module): The neural network model to be trained.
            patience (int): The number of epochs to wait for improvement in validation accuracy 
                            before triggering early stopping. Default is 2.

        Returns:
            torch.nn.Module: The trained neural network model.

        The training process logs training loss and validation accuracy using TensorBoard. The best model 
        based on validation accuracy is saved. The learning rate is adjusted if the validation accuracy 
        does not improve, and training is stopped early if there is no improvement in validation accuracy 
        for a given number of epochs specified by 'patience'.
        """
        optimizer = torch.optim.Adam(NeuralNet.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min')  # Learning rate scheduler
        early_stopping = EarlyStopping(patience=patience)  # Early stopping
        best_accuracy = 0.0


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

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(NeuralNet.state_dict(), f'./data/last_checkpoint_epoch.pt')

            scheduler.step(accuracy)  # Adjust learning rate based on validation accuracy
            early_stopping(accuracy, NeuralNet)  # Check early stopping condition

            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.writer.close()  # Close the TensorBoard writer
        return NeuralNet

class EarlyStopping:
    """
    A utility class for early stopping during the training of a neural network.

    This class is used to stop the training process if the validation metric 
    stops improving after a certain number of epochs. It is a form of regularization 
    used to prevent overfitting.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping the training.
        counter (int): Counter that keeps track of how many epochs have passed without improvement.
        best_score (float, optional): The best score achieved by the model so far. Default is None.
        early_stop (bool): Flag indicating whether early stopping criteria were met and training should stop.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(self, patience, delta=0):
        """
        Initializes the EarlyStopping instance.

        Args:
            patience (int): Number of epochs to wait for an improvement in the monitored metric.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_accuracy):
        """
        Call method to update the early stopping mechanism.

        This method should be called at the end of each epoch with the current 
        validation metric. It will compare the current score with the best score 
        and update the counter or reset it based on the improvement.

        Args:
            val_accuracy (float): The current validation accuracy of the model.
        """
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0