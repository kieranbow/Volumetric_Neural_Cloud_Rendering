#%% import libraries
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader, IterableDataset, Dataset

import pandas as pd

from torchtext.legacy import data
from torchtext.legacy import datasets

#%% Multi-Layered Perceptron
input_size = 3

class mlp(nn.Module):
    # init the mlp with 8 hidden layers and output
    def __init__(self):
        super().__init__()
        
        # Number of hidden nodes in each layer
        hidden_layer_1 = 3

        # number of outputs
        numberOfClasses = 1
        
        self.fc1 = nn.Linear(input_size, numberOfClasses)
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        
        return x;

#%% Init mlp model
model = mlp()
print(model)

#%% Load data for training
# https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
# https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00
# https://colab.research.google.com/github/bentrevett/pytorch-sentiment-analysis/blob/master/A%20-%20Using%20TorchText%20with%20Your%20Own%20Datasets.ipynb
# https://stackoverflow.com/questions/55109684/how-to-handle-large-json-file-in-pytorch
# https://www.youtube.com/watch?v=KRgq4VnCr7I
# https://stats.stackexchange.com/questions/161189/train-a-neural-network-to-distinguish-between-even-and-odd-numbers
# https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
# https://dzlab.github.io/dltips/en/pytorch/torchtext-datasets/

batch_size = 32

# =============================================================================
# class JsonDataset(IterableDataset):
#     def __init__(self, file):
#         self.file = file
#         
#     def __iter__(self):
#         #for json_file in self.file:
#             with open(self.file) as f:
#                 for sample_line in f:
#                     sample = json.loads(sample_line)
#                     yield sample['x'], sample ['y'], sample ['z']
# =============================================================================

#%%
#define field objects
X = data.Field()
Y = data.Field()
Z = data.Field()

# Define the dictionary which represents the dataset
fields = {'x': ('x', X), 'y': ('y', Y), 'z': ('z', Z)}

train_dataset, valid_dataset, test_dataset = data.TabularDataset.splits(
    path        = 'D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Data',
    train       = 'Training.json',
    validation  = 'Validation.json',
    test        = 'Testing.json',
    format      = 'json',
    fields      = fields)

# Print 5 elements from training dataset
#print("printing 5 elements from training dataset")
#for i in range(5):
    #print(vars(train_dataset[i]))

# Print 5 elements from validation dataset
#print("printing 5 elements from validation dataset")
#for i in range(5):
    #print(vars(valid_dataset[i]))
    
# Print 5 elements from testing dataset
#print("printing 5 elements from testing dataset")
#for i in range(5):
    #print(vars(test_dataset[i]))

#%%
#dataset = JsonDataset('D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Points.json')
#dataLoader = DataLoader(dataset, batch_size)

def collate_fn(batch):
    texts, labels = [], []
    
    for label, txt in batch:
        texts.append(txt)
        labels.append(label)
    return texts, labels

dataLoader = DataLoader(train_dataset, 4, shuffle=False)

#%% Train model
num_epochs = 100
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(num_epochs):
    loss = 0
    correct = 0

    model.train()
    for _data, target in dataLoader:
        optimizer.zero_grad()
        output = model(_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 0],
                            [0, 1, 1]])

training_output = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random: ")
print(synaptic_weights)

for iter in range(20000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    
    error = training_output - outputs
    adjustment = error * sigmoid_derivative(outputs)
    
    synaptic_weights += np.dot(input_layer.T, adjustment)

print("Synaptic weights after training: ")
print(synaptic_weights)

print("Outputs")
print(outputs)

