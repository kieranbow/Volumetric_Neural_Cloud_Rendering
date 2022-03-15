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

#%%
# A class that can read and iterate csv files
class CSVdataset(Dataset):
    def __init__(self, csv_filePath, transform = None):
        self.file = pd.read_csv(csv_filePath)
        self.transform = transform
        
    # Returns the length of the file
    def __len__(self):
        return len(self.file)
    
    # Returns the item from the file
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        x = self.file.iloc[index, 0]
        y = self.file.iloc[index, 1]
        z = self.file.iloc[index, 2]
        
        row = np.array([x,y,z])
        row = row.astype(float)
        sample = {'x' : x, 'y': y, 'z': z}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


training_data = CSVdataset('D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Data\\Training.csv')

for i in range(len(training_data)):
    sample = training_data[i]
    print(i, sample['x'], sample['y'], sample['z'])

#%%
dataloader = DataLoader(training_data, batch_size=4, shuffle=False)

for batch, sample_batch in enumerate(dataloader):
    print(batch, sample_batch['x'], sample_batch['y'], sample_batch['z'])

#%% Train model
# =============================================================================
# num_epochs = 100
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr = 0.001)
# 
# for epoch in range(num_epochs):
#     loss = 0
#     correct = 0
# 
#     model.train()
#     for _data, target in dataLoader:
#         optimizer.zero_grad()
#         output = model(_data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
# =============================================================================
    
#%%

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
 
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iter):
        for iter in range(training_iter):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid(output))
            self.synaptic_weights += adjustments
        
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output



neural_network = NeuralNetwork()

print("Random synatic weights ")
print(neural_network.synaptic_weights)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 0],
                            [0, 1, 1]])

training_output = np.array([[1, 1, 1, 1]]).T

neural_network.train(training_inputs, training_output, 20000)

print("Synaptic weights after training")
print(neural_network.synaptic_weights)
