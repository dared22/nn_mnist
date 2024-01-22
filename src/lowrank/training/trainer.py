import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lowrank.training.MNIST_downloader import Downloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lowrank.optimizers.SGD import SimpleSGD  
from tqdm import tqdm
from lowrank.optimizers.SGD import SimpleSGD
from lowrank.config_utils.config_parser import ConfigParser
from torchinfo import summary
from lowrank.optimizers.MultiOptim import MetaOptimizer

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
        early_stopping: Method checks if the validation loss does not improve beyond a certain threshold (min_delta) for a specified number of epochs (tolerance).
    """

    def __init__(self, model):
        """
        Initializes the Trainer class with the specified batch size and sets up data loaders for the MNIST dataset.

        The function also initializes a TensorBoard SummaryWriter to log training and validation metrics.

        Args:        """
        self.model = model
        self.numIterations = self.model.config_parser.num_epochs
        self.writer = SummaryWriter('./runs')  # TensorBoard SummaryWriter
        self.early_stopping_counter = 0
        self.accuracy = ()
        

    def train(self, train_dataloader, test_dataloader):
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
        optimizer = MetaOptimizer(self.model, self.model.config_parser.optimizer_config)
        criterion = nn.CrossEntropyLoss()
        # scheduler = ReduceLROnPlateau(optimizer, 'min')  # Learning rate scheduler
        best_accuracy = 0.0
        features = train_dataloader.dataset[0][0].shape[0]


        for epoch in range(self.numIterations):
            train_loss = 0.0
            self.model.train()  # Set the model to training mode

            train_loader_progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.numIterations}')

            for step, (images, labels) in enumerate(train_loader_progress):
                optimizer.zero_grad()
                out = self.model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        

                if (step + 1) % 100 == 0:
                    train_loader_progress.set_description(f'Epoch [{epoch+1}/{self.numIterations}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                    self.writer.add_scalar('Training Loss', loss.item(), epoch*len(train_dataloader) + step)
                
            train_loss /= len(train_dataloader) # Calculate average training loss for the epoch

            self.model.eval()  # Set the model to evaluation mode
            validation_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_dataloader:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            # Calculate average validation loss and accuracy
            validation_loss /= len(test_dataloader)
            accuracy = correct / total
            print(f'Epoch [{epoch+1}/{self.numIterations}], Validation Accuracy: {100 * accuracy:.2f}%, Validation Loss: {validation_loss:.4f}')
            self.writer.add_scalar('Validation Accuracy', accuracy, epoch)
            self.writer.add_scalar('Validation Loss', validation_loss, epoch)

            # Check if current model is the best
            if accuracy > best_accuracy:
                self.accuracy = (accuracy,(epoch+1))
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), f'./data/best_model_at_epoch_{epoch+1}.pt')

            # Early stopping check
            if self.early_stopping(train_loss, validation_loss, min_delta=0.0001, tolerance=3):
                print("Early stopping triggered at epoch:", epoch + 1)
                break


            # Adjust learning rate based on validation loss
            # scheduler.step(validation_loss)

        self.writer.close()  # Close the TensorBoard writer
        print(f'The best accuracy was achieved at epoch nr.{self.accuracy[1]} with validation accuracy {100*self.accuracy[0]:.2f}%')
        return self.model

    def early_stopping(self, train_loss, validation_loss, min_delta, tolerance):
        """
        Checks if early stopping criteria are met.
        Args:
            train_loss (float): Training loss of the current epoch.
            validation_loss (float): Validation loss of the current epoch.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            tolerance (int): The number of epochs with no improvement after which training will be stopped.
        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if (validation_loss - train_loss) > min_delta:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= tolerance:
                return True
        else:
            self.early_stopping_counter = 0  # Reset counter if improvement is observed

        return False
