from simplenn import SimpleNeuralNetwork
from trainer import Trainer

n1 = 28**2
n2 = 264
n3 = 10

NeuralNet = SimpleNeuralNetwork(n1, n2, n3)

trainer = Trainer(64)
trainer.train(10, 0.01 , NeuralNet)