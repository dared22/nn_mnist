import torch
from lowrank.training.neural_network import FeedForward
from lowrank.training.trainer import Trainer


NeuralNet = FeedForward.create_from_config("tests/data/config_ex_ffn.toml")

trainer = Trainer(64)
trained_nn = trainer.train(2, 0.01 , NeuralNet)
#save trained model


path =  './data/trained_model.pt'
NeuralNet.export_model(trained_nn,path)


