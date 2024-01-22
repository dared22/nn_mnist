from lowrank.training.trainer import Trainer
from lowrank.training.neural_network import FeedForward
from lowrank.predict import predict, show_image
import customtkinter as ctk
from tkinter import filedialog
import torch
import tkinter.scrolledtext as st
import sys
import threading
from PIL import Image, ImageOps
import io
import numpy as np
import os
import glob


class GUI:
    def __init__(self, app):
        self.app = app
        self.app.title("Enhanced GUI with CustomTkinter")

        # Configure grid layout (2 columns)
        self.app.columnconfigure(0, weight=1)
        self.app.columnconfigure(1, weight=1)

        # Initialize variables
        self._selected_filename = None

        # Create and place widgets
        self.create_widgets()
        # Create NeuralNet
        self._NeuralNet = FeedForward.create_from_config(self.browse_files("Select a configfile", (('toml files', '*.toml'), ('All files', '*.*'))))

    class TextRedirector(object):
        def __init__(self, widget):
            self.widget = widget

        def write(self, str):
            self.widget.insert(ctk.END, str)
            self.widget.see(ctk.END)
            self.widget.update_idletasks()

        def flush(self): #AttributeError: 'TextRedirector' object has no attribute 'flush'
            pass

    def create_widgets(self):
        # Create and place all the widgets
        self.label_file_explorer = ctk.CTkLabel(self.app, text="File Explorer", width=100, height=40, fg_color="gray", text_color="white")
        self.label_file_explorer.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

        self.save_btn = ctk.CTkButton(self.app, text="Save Trained Model", command=self.save_model)
        self.save_btn.grid(row=3, column=1, pady=10, padx=10, sticky="ew")

        self.button_training = ctk.CTkButton(self.app, text="Train NN", command=self.start_training_thread)
        self.button_training.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

        self.button_predict = ctk.CTkButton(self.app, text="Predict a Number from mnist", command=self.predict_btn)
        self.button_predict.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

        self.button_draw = ctk.CTkButton(self.app, text="Draw a number", command=self.open_drawing_window)
        self.button_draw.grid(row=2, column=1, pady=10, padx=10, sticky="ew")

        self.load_model_btn = ctk.CTkButton(self.app, text='Load Trained Model', command=self.load_model)
        self.load_model_btn.grid(row=3, column=0, pady=10, padx=10, sticky="ew")

        self.feature_btn = ctk.CTkButton(self.app, text='Leo`s Feature', command=self.feature)
        self.feature_btn.grid(row=1, column=1, pady=10, padx=10, sticky="ew")

        self.output_area = st.ScrolledText(self.app, height=10)
        self.output_area.grid(row=5, column=0, columnspan=2, padx=10, sticky="ew")

        # Redirect stdout to the output area
        sys.stdout = self.TextRedirector(self.output_area)

    def feature(self):
        nns = []
        def choose_folder():
            filename = filedialog.askdirectory(initialdir="/", title="Select a Folder")
            return filename
        def list_toml_files(directory):
            # Construct the search pattern
            pattern = os.path.join(directory, '*.toml')
            # List all files ending with .toml
            toml_files = glob.glob(pattern)
            return toml_files
        paths = list_toml_files(choose_folder())
        for path in paths:
            nn = FeedForward.create_from_config(path)
            nns.append(self.train_nn(nn))

    # Function to train the neural network
    def train_nn(self, nn):
        trainer = Trainer() 
        trained_nn = trainer.train(nn)
        self._NeuralNet = trained_nn
        return trained_nn

    # Function to start training in a separate thread
    def start_training_thread(self):
        training_thread = threading.Thread(target=self.train_nn, args=(self._NeuralNet,))
        training_thread.start()

    # Function to browse files
    def browse_files(self, title, type):
        filename = filedialog.askopenfilename(initialdir="/", title=title, filetypes=type)
        self.label_file_explorer.configure(text="File Opened: " + filename)
        self._selected_filename = filename
        return filename

    def predict_btn(self):
        # Function to handle the prediction logic
        def handle_predict():
            input_value = int(input_entry.get())  # Get the input value
            # You can add your prediction logic here using the input_value
            print(f"Predicting with input: {input_value}")
            pop_up.destroy()  # Close the pop-up window after prediction
                # Predict numbers from MNIST dataset
            prediction, label = predict(self._NeuralNet, input_value)
            print(f'Model predicts: {prediction} and the actual number is {label}')

        # Create a pop-up window
        pop_up = ctk.CTkToplevel(self.app)
        pop_up.title("Write an index from MNIST dataset")

        # Add an entry widget for input
        input_entry = ctk.CTkEntry(pop_up, width=200)
        input_entry.pack(pady=10, padx=10)

        # Add a button to trigger the prediction
        predict_button = ctk.CTkButton(pop_up, text="Predict", command=handle_predict)
        predict_button.pack(pady=10)

    def load_model(self):
        trained_model = self._NeuralNet
        self._NeuralNet.import_model(trained_model, self.browse_files("Select a model to load", (('PyTorch files', '*.pt'),('All files', '*.*')))) # Loading the trained weights into the model
        trained_model.eval()
        self._NeuralNet = trained_model
        
    def save_model(self):
        def choose_folder():
            filename = filedialog.askdirectory(initialdir="/", title="Select a Folder")
            return filename
        self._NeuralNet.export_model(self._NeuralNet, f'{choose_folder()}/trained_model.pt')


    def open_drawing_window(self):
        def start_paint(event):
            """Set the starting point for the line."""
            global last_x, last_y
            last_x, last_y = event.x, event.y

        def paint(event):
            """Draw the line on the canvas."""
            global last_x, last_y
            x, y = event.x, event.y
            canvas.create_line((last_x, last_y, x, y), width=15, fill='black')
            last_x, last_y = x, y

        # Create a pop-up window
        pop_up = ctk.CTkToplevel(self.app)
        pop_up.title("Draw a Number")

        # Create a canvas widget for drawing
        canvas = ctk.CTkCanvas(pop_up, width=280, height=280, bg='white')
        canvas.pack(pady=20, padx=20)

        def save_canvas():
            img = get_image_from_canvas(canvas, 28, 28)  # Resize to 28x28 for MNIST
            img = process_image(img)
            tensor = image_to_tensor(img)
            self._NeuralNet.eval()
            with torch.no_grad():
                output = self._NeuralNet(tensor)
                prediction = output.argmax()  # Getting the most likely class
            print(f'Model predicts: {prediction.item()}')
            canvas.delete("all")



        save_btn = ctk.CTkButton(pop_up,text='Predict', command=save_canvas, width=280)
        save_btn.pack(padx=20, pady=20)

        # Bind mouse events to the canvas
        canvas.bind('<Button-1>', start_paint)
        canvas.bind('<B1-Motion>', paint)


        def get_image_from_canvas(canvas, width, height):
            # Create a PIL image from the canvas content
            ps = canvas.postscript(colormode='color')
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            img = img.resize((width, height))
            return img

        def process_image(img):
            # Convert to grayscale
            img = img.convert('L')
            img = ImageOps.invert(img)
            # Convert PIL image to NumPy array
            img_array = np.array(img)
            # Normalize the image array to [0, 1]
            img_array = img_array / 255.0
            return img_array

        def image_to_tensor(img):
            # Convert PIL image to PyTorch tensor
            tensor = torch.tensor(img, dtype=torch.float32)
            tensor = tensor.unsqueeze(0)  # Add batch dimension, if necessary
            return tensor
        
    sys.stdout = sys.__stdout__ #get output back to console

