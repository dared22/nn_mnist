import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unittest.mock import Mock, create_autospec, patch
from lowrank.training.trainer import Trainer
from lowrank.optimizers.meta_optimizer import MetaOptimizer
from lowrank.config_utils.config_parser import ConfigParser

def test_model_assignment():
    model = Mock()
    trainer = Trainer(model=model, optimizer=Mock(), criterion=Mock())
    assert trainer.model is model
