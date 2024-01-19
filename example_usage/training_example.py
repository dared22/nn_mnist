import torch
from lowrank.training.neural_network import FeedForward
from lowrank.training.trainer import Trainer
from lowrank.training.MNIST_downloader import Downloader

NeuralNet = FeedForward.create_from_config("/Users/simenkrogstie/Project INF202/project-inf202/tests/data/config_ex_ffn.toml")

trainer = Trainer(64)
trained_nn = trainer.train(20, 0.01 , NeuralNet)
