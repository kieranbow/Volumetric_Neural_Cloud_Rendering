#%% import libraries
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader, IterableDataset

from torchtext.legacy import data
from torchtext.legacy import datasets

#%% Multi-Layered Perceptron
input_size = 2352

class mlp(nn.Module):
    # init the mlp with 8 hidden layers and output
    def __init__(self):
        super().__init__()
        
        # Number of hidden nodes in each layer
        # Same number of hidden layers as NeRF
        hidden_layer_1 = 256
        #hidden_layer_2 = 256
        #hidden_layer_3 = 256
        #hidden_layer_4 = 256
        #hidden_layer_5 = 256
        #hidden_layer_6 = 256
        #hidden_layer_7 = 256
        #hidden_layer_8 = 256

        # number of outputs
        numberOfClasses = 3
        
        self.fc1 = nn.Linear(input_size, hidden_layer_1)
        self.fc2 = nn.Linear(hidden_layer_1, numberOfClasses)
        #self.fc2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        #self.fc3 = nn.Linear(hidden_layer_2, hidden_layer_3)
        #self.fc3 = nn.Linear(hidden_layer_3, hidden_layer_4)
        #self.fc3 = nn.Linear(hidden_layer_4, hidden_layer_5)
        #self.fc3 = nn.Linear(hidden_layer_5, hidden_layer_6)
        #self.fc3 = nn.Linear(hidden_layer_6, hidden_layer_7)
        #self.fc3 = nn.Linear(hidden_layer_7, hidden_layer_8)
        #self.fc4 = nn.Linear(hidden_layer_8, numberOfClasses)
        
        self.dropout = nn.Dropout(p = 0.2)
        
    def forward(self, x):
        x = x.view(-1, input_size)
        
        # Add the hidden layers with a relu activiation function
        x = self.dropout(F.relu(self.fc1(x)))
        #x = self.dropout(F.relu(self.fc2(x)))
        #x = self.dropout(F.relu(self.fc3(x)))
        
        # Add the output layer with softmax
        #x = F.log_softmax(self.fc4(x), dim= 1)
        x = F.log_softmax(self.fc2(x), dim= 1)
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

class JsonDataset(IterableDataset):
    def __init__(self, file):
        self.file = file
        
    def __iter__(self):
        #for json_file in self.file:
            with open(self.file) as f:
                for sample_line in f:
                    sample = json.loads(sample_line)
                    yield sample['x'], sample ['y'], sample ['z']


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
print("printing 5 elements from training dataset")
for i in range(5):
    print(vars(train_dataset[i]))

# Print 5 elements from validation dataset
print("printing 5 elements from validation dataset")
for i in range(5):
    print(vars(valid_dataset[i]))
    
# Print 5 elements from testing dataset
print("printing 5 elements from testing dataset")
for i in range(5):
    print(vars(test_dataset[i]))

#%%

#list = []
#with open('D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Points.json') as f:
    #for obj in f:
       # line = json.loads(obj)
        #list.append(line)
        
#for line in list:
   # print(line["x"], line["y"], line["z"])

dataset = JsonDataset('D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Points.json')
dataLoader = DataLoader(dataset, batch_size)

#%% Shows the first five entires in the dataset
print("Showing the first five entires in the dataset")
dataIter = iter(dataset)
print(next(dataIter))
print(next(dataIter))
print(next(dataIter))
print(next(dataIter))
print(next(dataIter))

#%% Train model

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

num_epochs = 100
valid_loss_min = np.Inf
training_loss = []
valiation_loss = []

for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    
    #for batch in dataLoader:
        #y = model(batch)
    
    
    #model.train()
    
    #for train_data, target in dataLoader:
       # optimizer.zero_grad()
       # output = model(train_data)
        #loss = criterion(output, target)
       # loss.backward()
       # optimizer.step()
       # train_loss += loss.item() * train_data.size(0)
        
   # for valid_data, target in dataLoader:
       # output = model(valid_data)
       # loss = criterion(output, target)
       # valid_loss += loss.item() * valid_data.size(0)
    
    #train_loss = train_loss / len(dataLoader.sampler)
    #valid_loss = valid_loss / len(dataLoader.sampler)
    
   # training_loss.append(train_loss)
   # valiation_loss.append(valid_loss)
    
   # print('Epoch: {} \tTraining Loss: {:.2f} \tValidation Loss: {:.2f}'.format(
   # epoch + 1, 
   # train_loss,
   # valid_loss
   # ))
    
