from lowrank.training.trainer import Trainer
from lowrank.training.neural_network import FeedForward
from lowrank.training.MNIST_downloader import Downloader  # Ensure you have the correct import for Downloader
from lowrank.predict import predict, show_image
import customtkinter as ctk
from lowrank.training.gui import GUI

app = ctk.CTk()
gui = GUI(app)
app.mainloop()




#path =  './data/trained_model.pt'
#NeuralNet.export_model(trained_nn,path)
#
#
#
#
#
## Load the trained model
#trained_model = NeuralNet  # Creating an instance of the model
#NeuralNet.import_model(trained_model, path) # Loading the trained weights into the model
#trained_model.eval()  # Setting the model to evaluation mode
#
#
## Predict numbers from MNIST dataset
