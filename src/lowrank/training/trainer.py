import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lowrank.optimizers.MultiOptim import MetaOptimizer

class Trainer:
    """
    A class for training and evaluating neural network models.

    Attributes:
        model (torch.nn.Module): The neural network model to be trained.
        writer (SummaryWriter): TensorBoard SummaryWriter for logging training and validation metrics.
        best_accuracy (float): The highest validation accuracy achieved during training.
        training_log (list): Log of training metrics for each epoch.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (callable): Loss function for the model.
        num_epochs (int): Number of training epochs.
    """

    def __init__(self, model, optimizer, criterion, writer_dir='./runs'):
        """
        Initializes the Trainer class.

        Args:
            model (nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer for training the model.
            criterion (callable): The loss function used for training.
            writer_dir (str): Directory for storing TensorBoard logs.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = model.config_parser.num_epochs
        self.writer = SummaryWriter(writer_dir)
        self.best_accuracy = 0.0
        self.training_log = []

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, patience: int = 3):
        """
        Trains the model and evaluates its performance on the test dataset.

        Args:
            train_dataloader (DataLoader): The DataLoader for training data.
            test_dataloader (DataLoader): The DataLoader for test data.
            patience (int): Number of epochs to wait for improvement before stopping.

        Returns:
            (nn.Module, list): The trained model and the training log.
        """
        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(train_dataloader, epoch)
            val_accuracy, val_loss = self._validate_model(test_dataloader)

            self.writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
            self.writer.add_scalar('Validation Loss', val_loss, epoch)

            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_accuracy': val_accuracy,
                'val_loss': val_loss
            }
            self.training_log.append(epoch_log)

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                model_save_path = f'./best_model_epoch_{epoch+1}.pt'
                torch.save(self.model.state_dict(), model_save_path)

            if self._early_stopping(val_loss, patience):
                break

        self.writer.close()
        return self.model, self.training_log

    def _train_epoch(self, train_dataloader: DataLoader, epoch: int) -> float:
        """
        Conducts a single training epoch.

        Args:
            train_dataloader (DataLoader): The DataLoader for training data.
            epoch (int): The current epoch number.

        Returns:
            float: The average training loss for the epoch.
        """
        train_loss = 0.0
        for _, (images, labels) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')):
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_dataloader)

    def _validate_model(self, test_dataloader: DataLoader) -> (float, float):
        """
        Evaluates the model on the test dataset.

        Args:
            test_dataloader (DataLoader): The DataLoader for test data.

        Returns:
            (float, float): Validation accuracy and validation loss.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = correct / total
        return val_accuracy, val_loss / len(test_dataloader)

    def _early_stopping(self, validation_loss: float, patience: int) -> bool:
        """
        Checks if early stopping criteria are met.

        Args:
            validation_loss (float): The validation loss for the current epoch.
            patience (int): The number of epochs to wait for improvement before stopping.

        Returns:
            bool: True if early stopping criteria are met, False otherwise.
        """
        if validation_loss < self.best_accuracy - 0.01:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= patience:
                return True
        return False

    @staticmethod
    def create_from_model(model: nn.Module, writer_dir='./runs') -> 'Trainer':
        """
        Create a Trainer instance from an existing model. This model should have a config_parser attribute 
        (i.e. the model should have been created using the FeedForward.create_from_config method).

        Args:
            model (nn.Module): The neural network model.
            writer_dir (str): Directory for TensorBoard logs.

        Returns:
            Trainer: An instance of the Trainer class.
        """
        optimizer_config = model.config_parser.optimizer_config
        optimizer = MetaOptimizer(model, optimizer_config)
        criterion = nn.CrossEntropyLoss()
        return Trainer(model, optimizer, criterion, writer_dir=writer_dir)
