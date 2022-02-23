#%% import libraries
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

#%% Multi-Layered Perceptron
input_size = 2352

class mlp(nn.Module):
    # init the mlp with 3 hidden layers and output
    def __init__(self):
        super().__init__()
        
        # Number of hidden nodes in each layer
        hidden_layer_1 = 1024
        hidden_layer_2 = 512
        hidden_layer_3 = 256
        
        # number of outputs
        numberOfClasses = 3
        
        self.fc1 = nn.Linear(input_size, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, hidden_layer_3)
        self.fc4 = nn.Linear(hidden_layer_3, numberOfClasses)
        
        self.dropout = nn.Dropout(p = 0.2)
        
    def forward(self, x):
        x = x.view(-1, input_size)
        
        # Add the hidden layers with a relu activiation function
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        # Add the output layer with softmax
        x = F.log_softmax(self.fc4(x), dim= 1)
        return x;

#%% Init mlp model
model = mlp()
print(model)
