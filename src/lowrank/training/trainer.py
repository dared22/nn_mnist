import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lowrank.training.MNIST_downloader import Downloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lowrank.optimizers.MultiOptim import MetaOptimizer
from tqdm import tqdm
from torchinfo import summary

class Trainer:
    """
    A class for training and evaluating neural network models on the MNIST dataset.

    Attributes:
        model (torch.nn.Module): The neural network model to be trained.
        writer (SummaryWriter): TensorBoard SummaryWriter for logging training and validation metrics.
        best_accuracy (float): The highest validation accuracy achieved during training.
        early_stopping_counter (int): Counter for early stopping.
    """

    def __init__(self, model, writer_dir='./runs'):
        """
        Initializes the Trainer class with the specified model and sets up TensorBoard.
        """
        self.model = model
        self.num_epochs = self.model.config_parser.num_epochs
        self.writer = SummaryWriter(writer_dir) # For TensorBoard
        self.best_accuracy = 0.0
        self.early_stopping_counter = 0

    def train(self, train_dataloader, test_dataloader, patience=3):
        """
        Trains the model and evaluates its performance on the test dataset.
        """
        optimizer = MetaOptimizer(self.model, self.model.config_parser.optimizer_config)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            train_loss = self._train_epoch(train_dataloader, optimizer, criterion, epoch)
            val_accuracy, val_loss = self._validate_model(test_dataloader, criterion)

            # Log validation metrics and adjust learning rate
            self.writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
            self.writer.add_scalar('Validation Loss', val_loss, epoch)
            scheduler.step(val_loss)

            # Save the best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), f'./best_model_epoch_{epoch+1}.pt')

            # Early stopping
            if self._early_stopping(val_loss, patience):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        self.writer.close()  # Close the TensorBoard writer
        print(f"The best accuracy was achieved at epoch {epoch+1} with validation accuracy {100 * self.best_accuracy:.2f}%")
        return self.model

    def _train_epoch(self, train_dataloader, optimizer, criterion, epoch):
        train_loss = 0.0
        train_loader_progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{self.num_epochs}')

        for step, (images, labels) in enumerate(train_loader_progress):
            optimizer.zero_grad()
            out = self.model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (step + 1) % 100 == 0:
                train_loader_progress.set_description(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                self.writer.add_scalar('Training Loss', loss.item(), epoch * len(train_dataloader) + step)

        return train_loss / len(train_dataloader)

    def _validate_model(self, test_dataloader, criterion):
        self.model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_dataloader:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        return val_accuracy, val_loss / len(test_dataloader)

    def _early_stopping(self, validation_loss, patience):
        """
        Check if early stopping criteria are met based on validation loss.
        """
        if validation_loss < self.best_accuracy - 0.01:
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= patience:
                return True

        return False
