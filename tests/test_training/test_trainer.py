import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unittest.mock import Mock, create_autospec, patch
from lowrank.training.trainer import Trainer
from lowrank.optimizers.MultiOptim import MetaOptimizer
from lowrank.config_utils.config_parser import ConfigParser

def test_model_assignment():
    model = Mock()
    trainer = Trainer(model=model, optimizer=Mock(), criterion=Mock())
    assert trainer.model is model
    
@patch('your_module.tqdm')  # Mock tqdm to speed up the test
def test_train_epoch_loss_calculation(mock_tqdm):
    model = Mock()
    model.return_value = torch.tensor([[0.1, 0.9]])  # Mock model output
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataloader = [([torch.rand(1, 10)], [torch.tensor([1])])]  # Mock DataLoader

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion)
    train_loss = trainer._train_epoch(train_dataloader, 0)

    assert train_loss >= 0  # Basic check to ensure loss is computed


