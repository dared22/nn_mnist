from torch.optim import Optimizer
import torch
import torch.nn as nn
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.optimizers.DynamO import DynamicLowRankOptimizer
from lowrank.optimizers.SGD import SimpleSGD
from lowrank.layers.dense_layer import DenseLayer
from lowrank.training.neural_network import FeedForward

class MetaOptimizer():
    def __init__(self, model, optimizer_config = None, alternating_training = True):
        self.model = model
        self.layer_optimizers = {}
        self.default_optimizer_params = []
        self.default_optimizer = None
        self.alternating_training = alternating_training
        if alternating_training:
            self.train_only_S = False

        if optimizer_config is None:
            optimizer_config = {
                "default": (SimpleSGD, {'lr': 3e-4}),
                DynamicLowRankLayer: (DynamicLowRankOptimizer, {'lr': 3e-4}),
                # Other layer types and their optimizers (if any)
            }
        
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
                else:
                    self.layer_optimizers[name] = optimizer_class(layer.parameters(), **optimizer_args)

            else:
                # Check if the layer has parameters
                if any(layer.named_parameters()):
                    # print(layer)
                    self.default_optimizer_params.extend(layer.parameters())
            if len(self.default_optimizer_params) > 0 and self.default_optimizer is None:
                optimizer_class, optimizer_args = optimizer_config["default"]
                self.default_optimizer = optimizer_class(layer.parameters(), **optimizer_args)


    def step(self):
        """Perform an optimization step for each layer."""
        if self.alternating_training:
            self.alternating_step()
        else:
            self.standard_step()
            

                
    def alternating_step(self):
        if self.train_only_S:
            for key, optimizer in self.layer_optimizers.items():
                if isinstance(optimizer, DynamicLowRankOptimizer):
                    optimizer.defaults["only_S"] = True # Ensure that only S is updated
                    optimizer.step()

            self.toggle_only_S()

        else: 
            for key, optimizer in self.layer_optimizers.items():
                if isinstance(optimizer, DynamicLowRankOptimizer):
                    optimizer.defaults["only_S"] = False # Ensure that everything is updated
                optimizer.step()
        
            if self.default_optimizer:
                self.default_optimizer.step()
            
            self.toggle_only_S()
    
    def standard_step(self):
        for key, optimizer in self.layer_optimizers.items():
            optimizer.step()
        
        if self.default_optimizer:
            self.default_optimizer.step()
        

    def toggle_only_S(self):
        """Toggle whether only S is updated or not."""
        self.train_only_S = not self.train_only_S
        

    def zero_grad(self):
        """Clear all gradients in the model."""
        self.model.zero_grad()

   
# -------------- Example of usage ----------------
# Needs to be removed 
if __name__ == "__main__":
    model = FeedForward.create_from_config("tests/data/config_ex_ffn.toml")

    import torch.optim as optim

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

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Example usage
    optimizer_config = {
        "default": (torch.optim.Adam, {'lr': 0.005}),
        DynamicLowRankLayer: (DynamicLowRankOptimizer, {'lr': 0.1}),
        # Other layer types and their optimizers (if any)
    }


    meta_optimizer = MetaOptimizer(model, optimizer_config)
    for epochs in range(10):
        for i, batch in enumerate(train_loader):
            inputs, labels = batch
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            meta_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            meta_optimizer.step()
            
            # validation accuracy
            # if i % 100 == 0:
            #     print(f"loss: {loss}")
            if i % 100 == 0 and i != 0:
                print(f"loss: {loss}")
                correct = 0
                total = 0
                for batch in test_loader:
                    inputs, labels = batch
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print(f"Accuracy: {100 * correct / total}")

