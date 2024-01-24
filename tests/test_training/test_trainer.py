import pytest
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data.
import torch.utils.tensorboard.writer.

from lowrank.training.trainer import Trainer
from pathlib import Path

# A mock model to test the trainer
class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__
        self.fc = nn.Linear(784, 10)

        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
        

@pytest.fixture
def mock_trainer():
    return Trainer()

def test_data_loader_initialization(mock_trainer):
    assert isinstance(mock_trainer.trainloader, torch.utils.data.DataLoader)
    assert isinstance(mock_trainer.testloader, torch.utils.data.DataLoader)

def test_training_proscess(mock_trainer):
    model = MockModel()
    initial_state_dict = model.state_dict()
    trained_model = mock_trainer.train(model)
    trained_model_state_dict = trained_model.state_dict()
    assert not all(torch.equal(initial_state_dict[k], trained_model_state_dict[k]) for k in initial_state_dict)

def test_early_stopping_functionality(mock_trainer):
    # simulate conditions for early stopping
    mock_trainer.early_stopping_counter = 2
    assert not mock_trainer.early_stopping(0.1, 0.2, 0.01, 3)
    assert mock_trainer.early_stopping(0.1, 0.2, 0.01, 2)

def test_logging_with_tensorboard(mock_trainer):
    # Ensure TensorBoard SummarWriter is initialized
    assert isinstance(mock_trainer.writer, torch.utils.tensorboard.writer.SummaryWriter)

def test_model_saving(mock_trainer):
    model = MockModel()
    mock_trainer.train(model)
    saved_model_path = Path(f'./data/best_model_at_epoch_{mock_trainer.accuracy[1]}.pt')
    assert saved_model_path.exists()

def test_training_output(mock_trainer):
    model = MockModel()
    trained_model = mock_trainer.train(model)
    assert isinstance(trained_model, nn.Module)
