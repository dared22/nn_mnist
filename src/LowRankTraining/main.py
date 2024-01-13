from simplenn import SimpleNeuralNetwork
from trainer import Trainer
from import_export import DataMover
import matplotlib.pyplot as plt
from MNIST_downloader import Downloader
import torch

n1 = 28**2
n2 = 264
n3 = 10

NeuralNet = SimpleNeuralNetwork()

trainer = Trainer(64)
trained_nn = trainer.train(1, 0.01 , NeuralNet)
#save trained model


path =  './data/trained_model.pt'
data_mover = DataMover(path)
data_mover.export(trained_nn)

downloader = Downloader()
train, test = downloader.get_data()
model = data_mover.imprt(SimpleNeuralNetwork)



# get some data from the test loader
iterator = iter(test)
X, y = next(iterator)
print(X,y)



yHatNet = NeuralNet(X)
yHatModel1 = model(X)


# show that the results overlap
fig = plt.figure(figsize=(12, 5))
plt.plot(yHatNet[:, 5].detach(), 'b', label='Original')
plt.plot(yHatModel1[:,5].detach(), 'ro', label='Loaded')
plt.legend()
plt.xlabel('Stimulus index')
plt.ylabel('Model output for node "6"')
plt.xlim([1000, 1100])

plt.show()
     
