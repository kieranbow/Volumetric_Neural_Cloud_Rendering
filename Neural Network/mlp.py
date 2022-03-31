#%% import libraries
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pandas as pd

#%% Multi-Layered Perceptron
# =============================================================================
# input_size = 3 * 1
# 
# class mlp(nn.Module):
#     def __init__(self):
#         super().__init__()
# 
#         # Number of hidden nodes in each layer
#         hidden_layer_1 = 3
# 
#         # number of outputs
#         numberOfClasses = 1
#         
#         #self.fc1 = nn.Linear(input_size, numberOfClasses)
#         self.fc1= nn.AdaptiveMaxPool1d(input_size, numberOfClasses)
#         
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = F.sigmoid(self.fc1(x))
#         return x;
# 
# =============================================================================
#%% Init mlp model
# =============================================================================
# model = mlp()
# print(model)
# =============================================================================

#%% -----Loading and creating the dataset-----

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
        d = self.file.iloc[index, 3]
        
        row = np.array([x,y,z,d])
        row = row.astype(float)
        sample = {'x' : x, 'y': y, 'z': z, 'density': d}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


training_data   = CSVdataset('D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Data\\Training.csv')
validation_data = CSVdataset('D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Data\\Validation.csv')
testing_data    = CSVdataset('D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Data\\Testing.csv')

# =============================================================================
# for i in range(len(training_data)):
#      sample = training_data[i]
#      print(i, sample['x'], sample['y'], sample['z'], sample['density'])
# =============================================================================


#%%
batch_size = 1

class AI(nn.Module):
     def __init__(self):
         super().__init__()
 
         input_size = 3
         output_size = 1
         hiddenLayer_size = 3
         
         self.fc1 = nn.Linear(input_size, hiddenLayer_size)
         self.fc2 = nn.Linear(hiddenLayer_size, output_size)
         
     def forward(self, x):
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         return x;
 
model = AI()
print(model)

loss_func = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.002)

#%%
sample = np.array([0,0,0], dtype=float)
output = np.array([0], dtype=float)

def sampleFromDataset(index, data, sample, output_ref):
    sample[0] = data[index]['x']
    sample[1] = data[index]['y']
    sample[2] = data[index]['z']
    
    output_ref[0] = data[index]['density']
    
    return sample, output_ref

print(sampleFromDataset(0, training_data, sample, output))


#%%
num_epochs = 100

for i in range(num_epochs):
    sampleFromDataset(i, training_data, sample, output)
    pred = model(torch.autograd.Variable(torch.FloatTensor(sample)))
    loss = loss_func(pred, torch.autograd.Variable(torch.FloatTensor(output)))
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#%%
train_iter        = DataLoader(training_data, batch_size=3, shuffle=False)
validation_iter   = DataLoader(validation_data, batch_size=3, shuffle=False)
testing_iter      = DataLoader(testing_data, batch_size=3, shuffle=False)


#for batch, sample_batch in enumerate(train_iter):
#     print(batch, sample_batch['x'], sample_batch['y'], sample_batch['z'], sample_batch['density'])

#%% -----Train model-----
criterion  = nn.NLLLoss()
optimizer = optim.Adam(list(model.parameters()), lr=0.002)

loss_func = nn.NLLLoss()

n_epochs = 10

valid_loss_min = np.inf

train_losses, val_losses = [], []

for epoch in range(n_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    
    model.train()
    
#    for i, (x, y, z, d) in enumerate(train_iter):
#         optimizer.zero_grad()
#         output = model(i)

    for i, _data in enumerate(train_iter, 0):
        x, y, z, d = _data.values()
#        print(_data)
        optimizer.zero_grad()       
        output = model(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * _data.size(0)
        
