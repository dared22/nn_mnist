from torch.optim import Optimizer
import torch
import torch.nn as nn
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.optimizers.DynamO import DynamicLowRankOptimizer
from lowrank.optimizers.SGD import SimpleSGD

class MetaOptimizer():
    def __init__(self, model, optimizer_types):
        self.model = model
        self.layer_optimizers = {}
        
        for name, layer in model.named_modules():
            layer_type = type(layer)

            if layer_type in optimizer_types:
                optimizer_class, optimizer_args = optimizer_types[layer_type]

                # Check if the layer is of type DynamicLowRankLayer
                if isinstance(layer, DynamicLowRankLayer):
                    # Directly pass the parameters of the DynamicLowRankLayer
                    layer_params = [layer.U, layer.S, layer.V, layer.bias]
                    self.layer_optimizers[name] = optimizer_class(layer_params, **optimizer_args)
                else:
                    self.layer_optimizers[name] = optimizer_class(layer.parameters(), **optimizer_args)

    def step(self):
        """Perform an optimization step for each layer."""
        for name, optimizer in self.layer_optimizers.items():
            # print(f"Optimizing layer: {name}")  # Diagnostic print
            # print(f"Optimizer: {optimizer}")  # Diagnostic print
            optimizer.step()
        # # Take one step using SGD for every parameter S in the layers of type DynamicLowRankLayer
        # for name, layer in self.model.named_modules():
        #     meta_optimizer.zero_grad()
        #     loss.backward()

        #     if isinstance(layer, DynamicLowRankLayer):
        #         optimizer = SimpleSGD([layer.S], lr=0.00001)
        #         optimizer.step()

    def zero_grad(self):
        """Clear all gradients in the model."""
        self.model.zero_grad()

   
# -------------- Exzample of usage ----------------

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Dynamic low rank layer
        self.dynamic_layer1 = DynamicLowRankLayer(784, 250, 50, activation=nn.ReLU())
        self.dynamic_layer2 = DynamicLowRankLayer(250, 10, 5, activation=nn.ReLU())
    def forward(self, x):
        # print("Original shape:", x.shape)
        x = nn.Flatten()(x)
        # print("After flattening:", x.shape)
        x = self.dynamic_layer1(x)
        x = self.dynamic_layer2(x)
        return x

import torch.optim as optim
# Initialize the network
input_size = 784  # example for flattened 28x28 image
output_size = 10   # example for 10 classes
rank = 50          # arbitrary rank for the dynamic layer
model = MyNetwork()
batch_size = 64
lambda_lr = 0.00001

loss_function = nn.CrossEntropyLoss() # Adjust according to your task

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformations applied on each image => here just converting them to tensor and normalizing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the MNIST images
])

# Downloading and loading MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Creating the DataLoader for MNIST
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Example usage
optimizer_types = {
    DynamicLowRankLayer: (DynamicLowRankOptimizer, {'lr': 2e-4}),
    # Other layer types and their optimizers (if any)
}


meta_optimizer = MetaOptimizer(model, optimizer_types)

# Training loop
for i, batch in enumerate(train_loader):
    if i < 10:
        inputs, labels = batch
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        meta_optimizer.zero_grad()
        loss.backward()
        meta_optimizer.step()
        # print(f"loss: {loss}")