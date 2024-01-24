import customtkinter as ctk
from lowrank.training.gui import GUI
import argparse
import os
import torch
from lowrank.training.trainer import Trainer
from lowrank.training.neural_network import FeedForward
from lowrank.training.MNIST_downloader import Downloader
from lowrank.config_utils.config_parser import ConfigParser
from torch.utils.data import DataLoader

class NNTrainerCLI:
    def __init__(self, config_path=None):
        self.config_path = config_path or 'config.toml'
        self._trainloader = None
        self._testloader = None
        self._NeuralNet = None
        self.batchSize = None
        if config_path:
            self.load_config()
            self.download_data()

    def load_config(self):
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        configparser = ConfigParser(self.config_path)
        configparser.load_config()
        self.batchSize = configparser.batch_size

    def download_data(self):
        downloader = Downloader()
        traindataset, testdataset = downloader.get_data()
        self._trainloader = DataLoader(traindataset, batch_size=self.batchSize, shuffle=True)
        self._testloader = DataLoader(testdataset, batch_size=self.batchSize, shuffle=False)
        self._NeuralNet = FeedForward.create_from_config(self.config_path)

    def train_nn(self):
        trainer = Trainer.create_from_model(self._NeuralNet)
        trained_nn, training_log = trainer.train(self._trainloader, self._testloader)
        self._NeuralNet = trained_nn
        print("Training completed. Log: ", training_log)

    def train_all_from_folder(self, folder_path):
        models = FeedForward.mass_create_models(folder_path)
        for filename, model in models.items():
            self._NeuralNet = model

            self.batchSize = model.config_parser.batch_size
            self.download_data()

            trainer = Trainer.create_from_model(model)
            trained_model, training_log = trainer.train(self._trainloader, self._testloader)
            model_save_path = f'./data/mass_created_model_from_file_{filename}.pt'
            torch.save(trained_model.state_dict(), model_save_path)
            print(f'Finished training for model from file {filename}, training log: {training_log}')

            




def launch_gui():
    app = ctk.CTk()
    gui = GUI(app)
    app.mainloop()



if __name__ == "__main__":
    default_config_path = 'config.toml'  # Default configuration file path
    parser = argparse.ArgumentParser(description="Train a neural network with a specified configuration file.")
    parser.add_argument('--config', type=str, default=default_config_path, help=f"Path to the TOML configuration file (default: {default_config_path})")
    parser.add_argument('--gui', action='store_true', help="Launch the GUI instead of CLI")
    parser.add_argument('--train_all', type=str, help="Train all models from a specified folder")
    args = parser.parse_args()

    if args.gui:
        launch_gui()
    elif args.train_all:
        trainer_cli = NNTrainerCLI()
        trainer_cli.train_all_from_folder(args.train_all)
    else:
        try:
            nn_trainer = NNTrainerCLI(args.config)
            nn_trainer.train_nn()
        except FileNotFoundError as e:
            print(e)
            print("Please provide a valid configuration file.")





