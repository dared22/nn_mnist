import customtkinter as ctk
from lowrank.training.gui import GUI
from lowrank.training.MNIST_downloader import Downloader
from torch.utils.data import DataLoader
from lowrank.config_utils.config_parser import ConfigParser
from lowrank.training.neural_network import FeedForward
from lowrank.training.trainer import Trainer
#
#app = ctk.CTk()
#gui = GUI(app)
#app.mainloop()
#
#
config_file_path ='report_figures_code/config_files/config_rank_5.toml'
model_path = './best_model_epoch_1.pt'

downloader = Downloader()
traindataset, testdataset = downloader.get_data()
trainloader = DataLoader(traindataset, batch_size=64, shuffle=True)
testloader = DataLoader(testdataset, batch_size=64, shuffle=False)



NeuralNet = FeedForward.create_from_config(config_file_path)
#trainer = Trainer.create_from_model(NeuralNet)
#trained_nn = trainer.train(trainloader, testloader)
trained_model = NeuralNet

trained_model = NeuralNet.import_model(trained_model, model_path)
