from lowrank.training.trainer import Trainer
from lowrank.training.neural_network import FeedForward
from lowrank.training.MNIST_downloader import Downloader  # Ensure you have the correct import for Downloader
from lowrank.predict import predict, show_image

# Create an instance of the FeedForward neural network from configuration
NeuralNet = FeedForward.create_from_config("tests/data/config_ex_ffn.toml")


trainer = Trainer()
trained_nn = trainer.train(NeuralNet)
#save trained model


path =  './data/trained_model.pt'
NeuralNet.export_model(trained_nn,path)



















## Load the dataset
#downloader = Downloader()
#train, test = downloader.get_data()
#mnist_tensors = test.data
#mnist_labels = test.targets  # Assuming this is how you get the labels
#
#
## Load the trained model
#model_path = './data/trained_model.pt'
#trained_model = NeuralNet
#trained_model.load_state_dict(torch.load(model_path))  # Loading the trained weights into the model
#trained_model.eval()  # Setting the model to evaluation mode
#
#
## Predict numbers from MNIST dataset
#mnist_predictions = predict(trained_model, mnist_tensors)
#
## Outputting predictions and labels
#for idx, (prediction, label) in enumerate(zip(mnist_predictions, mnist_labels)):
#    print(f"Image {idx}: Predicted number is {prediction}, Actual Label: {label}")
#
#show_image(mnist_tensors[9946])
