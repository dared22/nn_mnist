import torch
from lowrank.training.neural_network import FeedForward
from lowrank.training.trainer import Trainer
from lowrank.training.import_export import DataMover
from lowrank.training.MNIST_downloader import Downloader

NeuralNet = FeedForward.create_from_config("tests/data/config_ex_ffn.toml")

trainer = Trainer(64)
trained_nn = trainer.train(20, 0.01 , NeuralNet)
#save trained model


path =  './data/trained_model.pt'
data_mover = DataMover(path)
data_mover.export(trained_nn)

