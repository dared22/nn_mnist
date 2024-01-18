from lowrank.training.trainer import Trainer
from lowrank.training.neural_network import FeedForward
from lowrank.training.MNIST_downloader import Downloader  # Ensure you have the correct import for Downloader
from lowrank.predict import predict, show_image
#import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import torch
import tkinter.scrolledtext as st
import sys
import threading

_selected_filename = None
nn = None

# Class to redirect stdout
class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(ctk.END, str)
        self.widget.see(ctk.END)

# Function to train the neural network
def train_nn(path):
    global nn
    NeuralNet = FeedForward.create_from_config(path)
    trainer = Trainer()
    trained_nn = trainer.train(NeuralNet)
    nn = trained_nn

# Function to start training in a separate thread
def start_training_thread(path):
    training_thread = threading.Thread(target=train_nn, args=(path))
    training_thread.start()

# Function to browse files
def browse_files():
    global _selected_filename
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File")
    label_file_explorer.configure(text="File Opened: " + filename)
    _selected_filename = filename


app = ctk.CTk()
app.title("Enhanced GUI with CustomTkinter")

label_file_explorer = ctk.CTkLabel(app, text="File Explorer", width=100, height=40, fg_color="gray", text_color="white")
label_file_explorer.pack(pady=10)

button_explore = ctk.CTkButton(app, text="Browse Files", command=browse_files)
button_explore.pack(pady=10)

button_training = ctk.CTkButton(app, text="Train NN", command=lambda: start_training_thread(_selected_filename))
button_training.pack(pady=10)

# ScrolledText widget for displaying output
output_area = st.ScrolledText(app, height=10)
output_area.pack(expand=True, fill='both', pady=10)

# Redirect stdout to the output area
sys.stdout = TextRedirector(output_area)

app.mainloop()

# Reset stdout when the app closes
sys.stdout = sys.__stdout__

#save trained model




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
#prediction, label = predict(trained_model, 9832)
#
#print(prediction, label)
#