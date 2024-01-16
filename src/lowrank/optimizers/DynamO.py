from torch.optim import Optimizer
import torch
import torch.nn as nn
from lowrank.layers.dynamic_low_rank import DynamicLowRankLayer
from lowrank.optimizers.SGD import SimpleSGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DynamicLowRankOptimizer(Optimizer):
    def __init__(self, params, lr=2e-4):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        U, S, V, bias = None, None, None, None  # Initialize variables
        for group in self.param_groups:
            U = group['params'][0]
            S = group['params'][1]
            V = group['params'][2]
            bias = group['params'][3]
        

        # Check if all variables are assigned
        if U is None or S is None or V is None or bias is None:
            raise ValueError("U, S, V, or bias not found in optimizer parameters.")

        # Update U, S, V using the existing gradients
        with torch.no_grad():
            print(S.grad)
            
            # 1. Update U
            K_old = U @ S
            grad_U = U.grad
            K_new = K_old - self.defaults["lr"] * grad_U @ S
            U_new, R = torch.linalg.qr(K_new, 'reduced')
            M = U_new.t() @ U

            U.data = U_new

            # 2. Update V
            L_old = V @ S.t()
            L_new = L_old - self.defaults["lr"] * V.grad @ S.t()
            V_new, _ = torch.linalg.qr(L_new, 'reduced')
            N = V_new.t() @ V # Doesn't work if I transpose V_new here? 
            V.data = V_new # Loss becomes nan if I don't use .data

            # 3. Update S
            S_tilde = M @ S @ N.t() # Doesn't work if I transpose N here?
            S_new = S_tilde - self.defaults["lr"] * S.grad
            S.data = S_new

            # 4. Update bias
            bias.data = bias - self.defaults["lr"] * bias.grad

        
# -------------- Exzample of usage ----------------




if __name__ == "__main__":
        
    class MyNetwork(nn.Module):
        def __init__(self):
            super(MyNetwork, self).__init__()
            self.dynamic_layer = DynamicLowRankLayer(input_size=784, output_size=10, rank=50)

        def forward(self, x):
            x = nn.Flatten()(x)
            x = self.dynamic_layer(x)
            return x
                    
    model = MyNetwork()

    # Define a loss function
    criterion = nn.CrossEntropyLoss()

    # Extract the parameters of the DynamicLowRankLayer
    dynamic_params = [model.dynamic_layer.U, model.dynamic_layer.S, model.dynamic_layer.V, model.dynamic_layer.bias]

    # Initialize the optimizer with the layer's parameters
    optimizer = DynamicLowRankOptimizer(dynamic_params, lr=2e-4)

    # Example input and target data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        # Training loop
    for epoch in range(1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()   # Zero the gradients
            output = model(data)    # Correct forward pass            
            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            print(f'Epoch {epoch}, Loss: {loss.item()}')