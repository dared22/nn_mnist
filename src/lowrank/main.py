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
input_value = None


class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str):
        self.widget.insert(ctk.END, str)
        self.widget.see(ctk.END)




# Function to train the neural network
def train_nn(path):
    NeuralNet = FeedForward.create_from_config(path)
    trainer = Trainer() 
    trained_nn = trainer.train(NeuralNet)
    global nn
    nn = trained_nn

# Function to start training in a separate thread
def start_training_thread():
    training_thread = threading.Thread(target=train_nn, args=(_selected_filename,))
    training_thread.start()

# Function to browse files
def browse_files():
    global _selected_filename
    filename = filedialog.askopenfilename(initialdir="/", title="Select a File")
    label_file_explorer.configure(text="File Opened: " + filename)
    _selected_filename = filename

def predict_btn():
    # Function to handle the prediction logic
    def handle_predict():
        global input_value
        input_value = input_entry.get()  # Get the input value
        # You can add your prediction logic here using the input_value
        print(f"Predicting with input: {input_value}")
        pop_up.destroy()  # Close the pop-up window after prediction

    # Create a pop-up window
    pop_up = ctk.CTkToplevel(app)
    pop_up.title("Write an index from MNIST dataset")

    # Add an entry widget for input
    input_entry = ctk.CTkEntry(pop_up, width=200)
    input_entry.pack(pady=10, padx=10)

    # Add a button to trigger the prediction
    predict_button = ctk.CTkButton(pop_up, text="Predict", command=handle_predict)
    predict_button.pack(pady=10)
    # Predict numbers from MNIST dataset
    prediction, label = predict(nn, input_value)
    print(prediction, label)






app = ctk.CTk()
app.title("Enhanced GUI with CustomTkinter")

# Configure grid layout (2 columns)
app.columnconfigure(0, weight=1)
app.columnconfigure(1, weight=1)

# File Explorer label
label_file_explorer = ctk.CTkLabel(app, text="File Explorer", width=100, height=40, fg_color="gray", text_color="white")
label_file_explorer.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

# Browse Files button
button_explore = ctk.CTkButton(app, text="Browse Files", command=browse_files)
button_explore.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

# Train NN button
button_training = ctk.CTkButton(app, text="Train NN", command=start_training_thread)
button_training.grid(row=1, column=1, pady=10, padx=10, sticky="ew")

# Predict a Number button
button_predict = ctk.CTkButton(app, text="Predict a Number from mnist", command=predict_btn)
button_predict.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

# Draw a Number button
button_draw = ctk.CTkButton(app, text="Draw a number")
button_draw.grid(row=2, column=1, pady=10, padx=10, sticky="ew")

# Output area
output_area = st.ScrolledText(app, height=10)
output_area.grid(row=4, column=0, columnspan=2, padx=10, sticky="ew")
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