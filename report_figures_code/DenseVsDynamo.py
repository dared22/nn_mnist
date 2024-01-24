from lowrank.training import FeedForward
from lowrank.training import Trainer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import toml

# ------------------ Utility functions ------------------

def params_in_dense(input, out):
	return input * out + out

def params_in_lowrank(input, out, rank):
	return input * rank + rank * rank + rank * out + out

def params_in_basedynamic(rank):
	layer_sizes = [(784, 256), (256, 128), (128, 64), (64, 10)]
	params = 0
	for x, y in layer_sizes[:-1]:
		params += params_in_lowrank(x, y, rank)
	params += params_in_dense(layer_sizes[-1][0], layer_sizes[-1][1])
	return params

total_params_dense = sum(params_in_dense(x,y) for x,y in [(784, 256), (256, 128), (128, 64), (64, 10)])
print(total_params_dense)

# ------------------ Data Prep ------------------
