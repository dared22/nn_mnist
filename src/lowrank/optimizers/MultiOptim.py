from torch.optim import Optimizer
import torch
import torch.nn as nn
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.optimizers.DynamO import DynamicLowRankOptimizer
from lowrank.optimizers.SGD import SimpleSGD

class MetaOptimizer():
    def __init__(self, model, optimizer_config = None):
        self.model = model
        self.layer_optimizers = {}
        self.s_params = [] # Collect S parameters from DynamicLowRankLayers
        self.default_optimizer_params = []
        self.default_optimizer = None
        
        for name, layer in model.named_modules():
            if layer == model: # Skip the model itself so it doesn't get assigned to default optimizer (which breaks everything)
                continue

            layer_type = type(layer)
            if layer_type in optimizer_config:
                optimizer_class, optimizer_args = optimizer_config[layer_type]

                # Check if the layer is of type DynamicLowRankLayer
                if isinstance(layer, DynamicLowRankLayer):
                    # Directly pass the parameters of the DynamicLowRankLayer for safety (optimizer doesn't know which parameters are which due to scope stuff, so if the parameters aren't passed in the right order, the optimizer *might* unpack them incorrectly and subsequently break)
                    layer_params = [layer.U, layer.S, layer.V, layer.bias]
                    self.layer_optimizers[name] = optimizer_class(layer_params, **optimizer_args)
                    self.s_params.append(layer.S)
                else:
                    self.layer_optimizers[name] = optimizer_class(layer.parameters(), **optimizer_args)

            else:
                # Check if the layer has parameters
                if any(layer.named_parameters()):
                    # print(layer)
                    self.default_optimizer_params.extend(layer.parameters())
            if len(self.default_optimizer_params) > 0:
                optimizer_class, optimizer_args = optimizer_config["default"]
                self.default_optimizer = optimizer_class(layer.parameters(), **optimizer_args)


    def step(self):
        """Perform an optimization step for each layer."""
        for key, optimizer in self.layer_optimizers.items():
            # print(f"Optimizing layer: {name}")  # Diagnostic print
            # print(f"Optimizer: {optimizer}")  # Diagnostic print
            optimizer.step()
        
        if self.default_optimizer:
            self.default_optimizer.step()

        # # Take one step using SGD for every parameter S in the layers of type DynamicLowRankLayer
        # Make one optimizer for parameter S of all layers of type DynamicLowRankLayer

    def zero_grad(self):
        """Clear all gradients in the model."""
        self.model.zero_grad()

   
# -------------- Example of usage ----------------
# Needs to be removed 
if __name__ == "__main__":
    class MyNetwork(nn.Module):
        def __init__(self):
            super(MyNetwork, self).__init__()
            # Dynamic low rank layer
            self.dynamic_layer1 = DynamicLowRankLayer(1200, 50, 30, activation=nn.ReLU())
            # self.linearlayer1 = nn.Linear(50, 50)
            self.dynamic_layer2 = DynamicLowRankLayer(50, 10, 5, activation=nn.ReLU())
            self.conv1 = nn.Conv2d(1, 3, 3)
            self.conv2 = nn.Conv2d(3, 3, 3)
            self.conv3 = nn.Conv2d(3, 3, 3)
            self.conv4 = nn.Conv2d(3, 3, 3)
        def forward(self, x):
            # print("Original shape:", x.shape)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = nn.Flatten()(x)

            # print("After flattening:", x.shape)
            x = self.dynamic_layer1(x)
            # x = self.linearlayer1(x)
            # x = nn.ReLU()(x)
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
    optimizer_config = {
        "default": (SimpleSGD, {'lr': 3e-4}),
        DynamicLowRankLayer: (DynamicLowRankOptimizer, {'lr': 3e-4}),
        # Other layer types and their optimizers (if any)
    }


    meta_optimizer = MetaOptimizer(model, optimizer_config)

    # Training loop
    for i, batch in enumerate(train_loader):
        if i < 1000:
            inputs, labels = batch
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            meta_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            meta_optimizer.step()
            print(f"loss: {loss}")