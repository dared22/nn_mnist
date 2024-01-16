import torch
import torch.nn as nn
import torch.nn.init as init

class DynamicLowRankLayer(nn.Module):
    def __init__(self, input_size, output_size, rank, activation=None):
        super(DynamicLowRankLayer, self).__init__()
        # Initialize U, S, and V
        if rank > min(input_size, output_size):
            raise ValueError("The rank cannot be larger than the minimum of input_size and output_size.")
        self.U = nn.Parameter(torch.randn(input_size, rank))
        self.S = nn.Parameter(torch.randn(rank, rank))
        self.V = nn.Parameter(torch.randn(output_size, rank))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.activation = None

    def forward(self, x):
        x = torch.matmul(x, self.U)
        x = torch.matmul(x, self.S)
        x = torch.matmul(x, self.V.t())
        x = x + self.bias
        if self.activation:
            x = self.activation(x)
        return x
    
    # def update_matrices(self, x, y, loss_function, lambda_lr):
    #     # Update U and V using dynamic low-rank training
    #     x = nn.Flatten()(x)

    #     # 1. Update U
    #     K_old = self.U @ self.S
    #     grad_U = torch.autograd.grad(loss_function(self.forward(x), y), self.U, retain_graph=True)[0]
    #     K_new = K_old - lambda_lr * grad_U @ self.S
    #     U_new, R = torch.qr(K_new)
    #     M = U_new.t() @ self.U

    #     self.U.data = U_new

    #     # 2. Update V
    #     L_old = self.V @ self.S.t()
    #     grad_V = torch.autograd.grad(loss_function(self.forward(x), y), self.V, retain_graph=True)[0]
    #     L_new = L_old - lambda_lr * grad_V @ self.S.t()
    #     V_new, _ = torch.qr(L_new.t())
    #     N = V_new @ self.V
    #     self.V.data = V_new.t()

    #     # 3. Update S
    #     S_tilde = M @ self.S @ N
    #     grad_S = torch.autograd.grad(loss_function(self.forward(x), y), self.S)[0]
    #     S_new = S_tilde - lambda_lr * grad_S
    #     self.S.data = S_new

    #     # 4. Update bias
    #     grad_bias = torch.autograd.grad(loss_function(self.forward(x), y), self.bias)[0]
    #     self.bias.data = self.bias - lambda_lr * grad_bias
    

# Test
if __name__ == "__main__":
    class MyNetwork(nn.Module):
        def __init__(self, input_size, output_size, rank):
            super(MyNetwork, self).__init__()
            # Dynamic low rank layer
            self.dynamic_layer = DynamicLowRankLayer(input_size, output_size, rank, activation=nn.ReLU())

        def forward(self, x):
                print("Original shape:", x.shape)
                x = nn.Flatten()(x)
                print("After flattening:", x.shape)
                x = self.dynamic_layer(x)
                return x
    
    import torch.optim as optim
    # Initialize the network
    input_size = 784  # example for flattened 28x28 image
    output_size = 10   # example for 10 classes
    rank = 50          # arbitrary rank for the dynamic layer
    model = MyNetwork(input_size, output_size, rank)
    batch_size = 64
    lambda_lr = 0.003

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


# for i, batch in enumerate(train_loader):
#     if i < 50:
#         inputs, labels = batch

#         # Forward pass
#         outputs = model(inputs)
#         loss = loss_function(outputs, labels)

#         # Now update the DynamicLowRankLayer separately
#         model.dynamic_layer.update_matrices(inputs, labels, loss_function, lambda_lr)
#         print(loss)
#     else:
#         break

# # Zero the parameter gradients
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# for batch in train_loader: # Assuming train_loader is a DataLoader instance
#     inputs, labels = batch
#     optimizer.zero_grad()

#     # Forward pass
#     outputs = model(inputs)
#     loss = loss_function(outputs, labels)

#     # Backward pass and optimize
#     loss.backward()
#     optimizer.step()

#     # Now update the DynamicLowRankLayer separately
#     model.dynamic_layer.update_matrices(inputs, labels, loss_function, lambda_lr)

