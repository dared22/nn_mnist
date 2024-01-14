import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, layers):
        super(FeedForward, self).__init__()

        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X
    
# @staticmethod skal ta en path som input, og instanciate en config_parser
# for så å kjøre config_parer.create_ffn
# da vil config_parseren til å ha paramtere inni seg
# da skal jeg hente ut self.layers, en liste med layers jeg skal loope over og putte inn 
# i FeedForward klassen

    
    
