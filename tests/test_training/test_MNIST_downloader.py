import pytest
import os
import torch
from torchvision import datasets
from lowrank.training.MNIST_downloader import Downloader

def test_initialization_and_data_download():
    downloader = Downloader()
    assert isinstance(downloader.mnist_train, datasets.MNIST)
    assert isinstance(downloader.mnist_test, datasets.MNIST)

def test_data_transformation():
    downloader = Downloader()
    train, test = downloader.get_data()
    assert isinstance(train[0][0], torch.Tensor)
    assert isinstance(test[0][0], torch.Tensor)

def test_saved_data_files():
    downloader = Downloader()
    assert os.path.exists('./data/mnist_train.pt')
    assert os.path.exists('./data/mnist_test.pt')

def test_get_data_method():
    downloader = Downloader()
    train, test = downloader.get_data()
    assert isinstance(train, datasets.MNIST)
    assert isinstance(test, datasets.MNIST)

def test_data_integrity():
    downloader =Downloader()
    train, test = downloader.get_data()

    # Check if the datasets have the expected number of samples
    assert len(train) == 60000
    assert len(test) == 10000

    # Check a few sample labels to ensure they're integers
    assert isinstance(train[0][1], int) 
    assert isinstance(test[0][1], int)
    
