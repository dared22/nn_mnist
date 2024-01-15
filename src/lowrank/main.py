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

#downloader = Downloader()
#train, test = downloader.get_data()
#model = data_mover.imprt(SimpleNeuralNetwork)
#
#
#
## get some data from the test loader
#iterator = iter(test)
#X, y = next(iterator)
#
#NeuralNet.eval()
#model.eval()
#
#yHatNet = NeuralNet(X)
#yHatModel1 = model(X)
#
#
## show that the results overlap
#fig = plt.figure(figsize=(12, 5))
#plt.plot(yHatNet[:, 5].detach(), 'b', label='Original')
#plt.plot(yHatModel1[:,5].detach(), 'ro', label='Loaded')
#plt.legend()
#plt.xlabel('Stimulus index')
#plt.ylabel('Model output for node "6"')
#
#
#plt.show()
     
