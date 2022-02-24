#%% import libraries
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

#%% Multi-Layered Perceptron
input_size = 2352

class mlp(nn.Module):
    # init the mlp with 8 hidden layers and output
    def __init__(self):
        super().__init__()
        
        # Number of hidden nodes in each layer
        # Same number of hidden layers as NeRF
        hidden_layer_1 = 256
        hidden_layer_2 = 256
        hidden_layer_3 = 256
        hidden_layer_4 = 256
        hidden_layer_5 = 256
        hidden_layer_6 = 256
        hidden_layer_7 = 256
        hidden_layer_8 = 256

        # number of outputs
        numberOfClasses = 3
        
        self.fc1 = nn.Linear(input_size, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.fc3 = nn.Linear(hidden_layer_2, hidden_layer_3)
        self.fc3 = nn.Linear(hidden_layer_3, hidden_layer_4)
        self.fc3 = nn.Linear(hidden_layer_4, hidden_layer_5)
        self.fc3 = nn.Linear(hidden_layer_5, hidden_layer_6)
        self.fc3 = nn.Linear(hidden_layer_6, hidden_layer_7)
        self.fc3 = nn.Linear(hidden_layer_7, hidden_layer_8)
        self.fc4 = nn.Linear(hidden_layer_8, numberOfClasses)
        
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

#%% Load data for training
batch_size = 4

# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
# https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb
# use .json intead of .txt



#%% Train model
num_epochs = 100
valid_loss_min = np.Inf
training_loss = []
valiation_loss = []

for epoch in range(num_epochs):
    training_loss = 0.0
    validation_loss = 0.0
    
    model.train()
    
    
    
