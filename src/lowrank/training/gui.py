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
from torch.utils.data import DataLoader
from lowrank.training.MNIST_downloader import Downloader
from lowrank.config_utils.config_parser import ConfigParser


class GUI:
    """
    Graphical User Interface for a neural network training and prediction application.

    Parameters
    ----------
    app : tkinter.Tk or customtkinter.CTk
        The main application window for the GUI.

    Attributes
    ----------
    _selected_filename : str
        Path of the selected file.
    batchSize : int
        Batch size for training, read from the configuration file.
    testdataset : Dataset
        The test dataset used for prediction.
    _trainloader : DataLoader
        DataLoader for training data.
    _testloader : DataLoader
        DataLoader for test data.
    _NeuralNet : nn.Module
        The neural network model.
    """

    def __init__(self, app):
        """
        Initializes the GUI with necessary widgets and configurations.

        Parameters
        ----------
        app : tkinter.Tk or customtkinter.CTk
            The root or main window of the application where widgets are placed.
        """
        self.app = app
        self.app.title("Enhanced GUI with CustomTkinter")

        # Configure grid layout (2 columns)
        self.app.columnconfigure(0, weight=1)
        self.app.columnconfigure(1, weight=1)

        # Create and place widgets
        self.create_widgets()
        # Initialize variables
        self._selected_filename = None
        config_file_path = os.path.join(os.getcwd(), 'config.toml')

        # Check if the file exists as it 
        if os.path.isfile(config_file_path):
            pass
        else:
            config_file_path = self.browse_files("Select a config file", (('Config Files', '*.toml'),('All files', '*.*')))
        configparser = ConfigParser(config_file_path)
        configparser.load_config()
        self.batchSize = configparser.batch_size

        downloader = Downloader()
        traindataset, self.testdataset = downloader.get_data()
        self._trainloader = DataLoader(traindataset, batch_size=self.batchSize, shuffle=True)
        self._testloader = DataLoader(self.testdataset, batch_size=self.batchSize, shuffle=False)

        # Create NeuralNet
        self._NeuralNet = FeedForward.create_from_config(config_file_path)


    class TextRedirector(object):
        """
        Redirects the standard output to a tkinter widget.

        Attributes
        ----------
        widget : tkinter.Text or customtkinter.CTkScrolledText
            The tkinter widget where the output will be redirected.
        """
        def __init__(self, widget):
            """
            Parameters
            ----------
            widget : tkinter.Text or customtkinter.CTkScrolledText
                The tkinter widget where the output will be redirected.
            """
            self.widget = widget

        def write(self, str):
            """
            Write the string to the widget.

            Parameters
            ----------
            str : str
                The string to be written to the widget.
            """
            self.widget.insert(ctk.END, str)
            self.widget.see(ctk.END)
            self.widget.update_idletasks()

        def flush(self): 
            """
            Flush the widget's content. This method is a placeholder to comply with 
            standard output's interface.
            """
            pass

    def create_widgets(self):
        """
        Creates and places widgets in the main application window.
        """
        self.label_file_explorer = ctk.CTkLabel(self.app, text="File Explorer", width=100, height=40, fg_color="gray", text_color="white")
        self.label_file_explorer.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

        self.save_btn = ctk.CTkButton(self.app, text="Save Trained Model", command=self.save_model)
        self.save_btn.grid(row=3, column=1, pady=10, padx=10, sticky="ew")

        self.button_training = ctk.CTkButton(self.app, text="Train NN", command=lambda:self.start_training_thread(self._NeuralNet))
        self.button_training.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

        self.button_predict = ctk.CTkButton(self.app, text="Predict a Number from mnist", command=self.predict_btn)
        self.button_predict.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

        self.button_draw = ctk.CTkButton(self.app, text="Draw a number", command=self.open_drawing_window)
        self.button_draw.grid(row=2, column=1, pady=10, padx=10, sticky="ew")

        self.load_model_btn = ctk.CTkButton(self.app, text='Load Trained Model', command=self.load_model)
        self.load_model_btn.grid(row=3, column=0, pady=10, padx=10, sticky="ew")

        self.feature_btn = ctk.CTkButton(self.app, text='Train all from folder', command=self.train_all)
        self.feature_btn.grid(row=1, column=1, pady=10, padx=10, sticky="ew")

        self.output_area = st.ScrolledText(self.app, height=10)
        self.output_area.grid(row=5, column=0, columnspan=2, padx=10, sticky="ew")

        # Redirect stdout to the output area
        sys.stdout = self.TextRedirector(self.output_area)

    def train_all(self):
        """
        Trains neural networks using configurations from all TOML files in a selected folder.
        """
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
        for  path in paths:
            nn = FeedForward.create_from_config(path)
            trained_nn, training_log = self.train_nn(nn)
            model_save_path = f'./data/trained_model_from_file_{path[-11:-4]}.pt'
            trained_nn.export_model(trained_nn, model_save_path)
            nns.append((training_log,path[-11:-4])) #cut out rest of the path for visibility
        print(nns)
            


    def train_nn(self, nn):
        """
        Trains a given neural network model.

        Parameters
        ----------
        nn : nn.Module
            The neural network model to be trained.

        Returns
        -------
        tuple
            A tuple containing the trained neural network model and the training log.
        """
        trainer = Trainer.create_from_model(nn) 
        trained_nn, training_log = trainer.train(self._trainloader, self._testloader)
        self._NeuralNet = trained_nn
        return trained_nn, training_log

    def start_training_thread(self, nn):
        """
        Starts the training process of a neural network model in a separate thread.

        Parameters
        ----------
        nn : nn.Module
            The neural network model to be trained.
        """
        training_thread = threading.Thread(target=self.train_nn, args=(nn,))
        training_thread.start()


    def browse_files(self, title, type):
        """
        Opens a file dialog to browse and select files.

        Parameters
        ----------
        title : str
            The title of the file dialog.
        filetypes : tuple
            The filetype filter for the file dialog.

        Returns
        -------
        str
            The path of the selected file, or None if no file is selected.
        """
        filename = filedialog.askopenfilename(initialdir="/", title=title, filetypes=type)
        self.label_file_explorer.configure(text="File Opened: " + filename)
        self._selected_filename = filename
        return filename

    def predict_btn(self):
        """
        Handles the prediction process using a GUI pop-up for input.
        """
        def handle_predict():
            input_value = int(input_entry.get())  # Get the input value
            # You can add your prediction logic here using the input_value
            print(f"Predicting with input: {input_value}")
            pop_up.destroy()  # Close the pop-up window after prediction
                # Predict numbers from MNIST dataset
            prediction, label = predict(self._NeuralNet, input_value, self.testdataset)
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
        """
        Loads a trained neural network model from a selected file.
        """
        trained_model = self._NeuralNet
        self._NeuralNet.import_model(trained_model, self.browse_files("Select a model to load", (('PyTorch files', '*.pt'),('All files', '*.*')))) # Loading the trained weights into the model
        trained_model.eval()
        self._NeuralNet = trained_model
        
    def save_model(self):
        """
        Saves the currently loaded neural network model to a selected directory.
        """
        def choose_folder():
            filename = filedialog.askdirectory(initialdir="/", title="Select a Folder")
            return filename
        self._NeuralNet.export_model(self._NeuralNet, f'{choose_folder()}/trained_model.pt')


    def open_drawing_window(self):
        """
        Opens a pop-up window with a canvas for drawing digits to be predicted.
        """
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
            """
            Extracts an image from a canvas, resizes it, and returns the image.

            Parameters
            ----------
            canvas : ctk.CTkCanvas
                The canvas from which to capture the image.
            width : int
                The width to resize the image to.
            height : int
                The height to resize the image to.

            Returns
            -------
            PIL.Image
                The extracted and resized image.
            """
            ps = canvas.postscript(colormode='color')
            img = Image.open(io.BytesIO(ps.encode('utf-8'))) # Create a PIL image from the canvas content
            img = img.resize((width, height))
            return img

        def process_image(img):
            """
            Processes the image by converting it to grayscale and normalizing.

            Parameters
            ----------
            img : PIL.Image
                The image to be processed.

            Returns
            -------
            ndarray
                The processed image array.
            """
            img = img.convert('L')# Convert to grayscale
            img = ImageOps.invert(img) # Convert PIL image to NumPy array
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize the image array 
            return img_array

        def image_to_tensor(img):
            """
            Converts an image to a PyTorch tensor.
        
            Parameters
            ----------
            img : ndarray
                The image to be converted.
        
            Returns
            -------
            Tensor
                The image converted to a PyTorch tensor.
            """
            tensor = torch.tensor(img, dtype=torch.float32)
            tensor = tensor.unsqueeze(0) 
            return tensor
        
    sys.stdout = sys.__stdout__ #get output back to console

