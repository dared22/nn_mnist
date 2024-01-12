import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from MNIST_downloader import Downloader

class Trainer:
     def __init__(self, batch_size) -> None:     
        self.batchSize = batch_size

        downloader = Downloader()
        traindataset, testdataset = downloader.get_data()
        self.trainloader = DataLoader(traindataset, batch_size=self.batchSize, shuffle=True)
        self.testloader = DataLoader(testdataset, batch_size=self.batchSize, shuffle=False)
    
     def train(self, numIterations, lr, NeuralNet):
        # Number of iterations
        numIterations = numIterations
        lr = lr
        optimizer = torch.optim.Adam(NeuralNet.parameters(), lr=lr) #temporary
        
        for i in range(numIterations):
            for step, (images, labels) in enumerate(self.trainloader):
                optimizer.zero_grad()  # Reset gradients to zero
                out = NeuralNet(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
        
                if (step + 1) % 100 == 0:
                    print(f'Epoch [{i+1}/{numIterations}], Step [{step+1}/{len(self.trainloader)}], Loss: {loss.item():.4f}')
        
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.testloader:
                    outputs = NeuralNet(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
            accuracy = correct / total
            print(f'Epoch [{i+1}/{numIterations}], Validation Accuracy: {100 * accuracy:.2f}%')
     
