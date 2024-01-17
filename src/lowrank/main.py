from lowrank.training.trainer import Trainer
from lowrank.training.neural_network import FeedForward
from lowrank.training.MNIST_downloader import Downloader  # Ensure you have the correct import for Downloader
from lowrank.predict import predict, show_image
import torch

# Create an instance of the FeedForward neural network from configuration
NeuralNet = FeedForward.create_from_config("tests/data/config_ex_ffn.toml")


trainer = Trainer()
trained_nn = trainer.train(NeuralNet)
#save trained model
#
#
path =  './data/trained_model.pt'
#NeuralNet.export_model(trained_nn,path)





# Load the trained model
trained_model = NeuralNet  # Creating an instance of the model
NeuralNet.import_model(trained_model, path) # Loading the trained weights into the model
trained_model.eval()  # Setting the model to evaluation mode


# Predict numbers from MNIST dataset
prediction, label = predict(trained_model, 99)

print(prediction, label)
## Outputting predictions and labels
#for idx, (prediction, label) in enumerate(zip(mnist_predictions, mnist_labels)):
#    print(f"Image {idx}: Predicted number is {prediction}, Actual Label: {label}")
#
##show_image(mnist_tensors[9946])
