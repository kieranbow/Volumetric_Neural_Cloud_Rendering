#%% import libraries
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset

import pandas as pd

import random

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

# Randomly sample a line in the dataset and extract x, y, z, density 
# and return it
def sampleFromDataset(data, sample, output_ref):
    idx = random.randrange(len(data))
    sample[0]       = data[idx]['x']
    sample[1]       = data[idx]['y']
    sample[2]       = data[idx]['z']
    output_ref[0]   = data[idx]['density']
    
    return sample, output_ref

print(sampleFromDataset(training_data, sample, output))


#%%
num_epochs = 1000

for i in range(num_epochs):
    sampleFromDataset(training_data, sample, output)
    pred = model(torch.autograd.Variable(torch.FloatTensor(sample)))
    loss = loss_func(pred, torch.autograd.Variable(torch.FloatTensor(output)))
    #print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#%%
def writeLayer(weights, name, file):
    if len(list(weights[name].shape)) == 2:
        for y in range(weights[name].shape[0]):
            line = ""
            for x in range(weights[name].shape[1]):
                line = line + " " + str(float(weights[name][y][x]))
                line = line + "\n"
                file.write(line)
    else:
        line = ""
        for x in range(weights[name].shape[0]):
            line = line + " " + str(float(weights[name][x]))
            line = line + "\n"
            file.write(line)

def write(weights, file):
    keys = weights.keys()
    
    for k in keys:
        writeLayer(weights, k, file) 
        
weights = model.state_dict()

file = open("D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Weights.txt", "w")
write(weights, file)
file.close()