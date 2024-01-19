# It's very easy to go from an existing model to a model with dynamic low-rank layers.
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.optimizers.MultiOptim import MetaOptimizer
from lowrank.optimizers.DynamO import DynamicLowRankOptimizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Say you have a model that looks like this:
class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.c1 = nn.Conv2d(1, 5, 5)
		self.c2 = nn.Conv2d(5, 10, 5)
		self.c3 = nn.Conv2d(10, 15, 5)
		
		self.fc1 = nn.Linear(3840, 100)
		self.fc2 = nn.Linear(100, 10)

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		x = nn.Flatten()(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return x
	
# Trains like this:
model = Network()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 1

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(epochs):
	for i, data in enumerate(train_loader):
		if i > 10:
			break
		X, y = data
		model.zero_grad()
		output = model(X)
		loss = criterion(output, y)
		loss.backward()
		optimizer.step()
		print(loss)

# Now, we want to add dynamic low-rank layers to the model. We do this by replacing the Dense layers with dynamic low-rank layers.
class DynamicNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		# self.c1 = nn.Conv2d(1, 5, 5)
		# self.c2 = nn.Conv2d(5, 10, 5)
		# self.c3 = nn.Conv2d(10, 15, 5)
		
		self.fc1 = DynamicLowRankLayer(784, 100, 30, activation=nn.ReLU())
		self.fc2 = DynamicLowRankLayer(100, 10, 8)

	def forward(self, x):
		# x = self.c1(x)
		# x = nn.ReLU()(x)
		# x = self.c2(x)
		# x = nn.ReLU()(x)
		# x = self.c3(x)
		# x = nn.ReLU()(x)
		x = nn.Flatten()(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return x
	
# For the training loop, the only difference is that we need to use the MetaOptimizer instead of the regular optimizer.
model = DynamicNetwork()
optimizer_config = {
	"default": (torch.optim.Adam, {'lr': 3e-4}),
	DynamicLowRankLayer: (DynamicLowRankOptimizer, {'lr': 3e-4}),
}

meta_optimizer = MetaOptimizer(model, optimizer_config)

for epoch in range(epochs):
	for i, data in enumerate(train_loader):
		if i > 1000:
			break
		X, y = data
		model.zero_grad()
		output = model(X)
		loss = criterion(output, y)
		loss.backward()
		meta_optimizer.step()
		# calulate accuracy

		