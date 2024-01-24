import pytest
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.tensorboard.writer as SummaryWriter
from lowrank.training.trainer import Trainer
from pathlib import Path
from lowrank.training.neural_network import FeedForward
from lowrank.training.MNIST_downloader import Downloader


#Download data
downloader = Downloader()
train, test = downloader.get_data()       


@pytest.fixture
def mock_trainer():
    model = FeedForward.create_from_config('./report_figures_code/config_files/config_rank_5.toml')
    return Trainer.create_from_model(model)
#trainloader
@pytest.fixture
def train_dataloader():
    train_dataset = train
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader
#testloader
@pytest.fixture
def test_dataloader():
    test_dataset = test 
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader


#Test Training Process
def test_training_process(mock_trainer, train_dataloader, test_dataloader):
    initial_state_dict = mock_trainer.model.state_dict()
    trained_model, _ = mock_trainer.train(train_dataloader, test_dataloader)
    trained_model_state_dict = trained_model.state_dict()
    assert not all(torch.equal(initial_state_dict[k], trained_model_state_dict[k]) for k in initial_state_dict)

#Test Early Stopping Functionality
def test_early_stopping_functionality(mock_trainer):
    mock_trainer.early_stopping_counter = 2
    stopped = mock_trainer._early_stopping(0.2, 3) 
    assert stopped, "Early stopping did not trigger as expected"

#Test Logging with TensorBoard
def test_logging_with_tensorboard(mock_trainer):
    assert isinstance(mock_trainer.writer, torch.utils.tensorboard.writer.SummaryWriter)

#Test Model Saving
def test_model_saving(mock_trainer, train_dataloader, test_dataloader):
    trained_model, _ = mock_trainer.train(train_dataloader, test_dataloader)
    saved_model_path = Path(f'./data/best_model_epoch_{mock_trainer.num_epochs}.pt')
    assert saved_model_path.exists()

#Test Training Output
def test_training_output(mock_trainer, train_dataloader, test_dataloader):
    trained_model, training_log = mock_trainer.train(train_dataloader, test_dataloader)
    assert isinstance(trained_model, nn.Module)
    assert isinstance(training_log, list)