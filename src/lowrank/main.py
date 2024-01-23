import customtkinter as ctk
from lowrank.training.gui import GUI
from lowrank.training.MNIST_downloader import Downloader
from torch.utils.data import DataLoader
from lowrank.config_utils.config_parser import ConfigParser
from lowrank.training.neural_network import FeedForward
from lowrank.training.trainer import Trainer
#
app = ctk.CTk()
gui = GUI(app)
app.mainloop()

