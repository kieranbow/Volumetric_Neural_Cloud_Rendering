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
class AI(nn.Module):
     def __init__(self):
         super().__init__()
 
         input_size = 3
         output_size = 1
         hiddenLayer_size = 3
         
         self.fc1 = nn.Linear(input_size, hiddenLayer_size)
         self.fc2 = nn.Linear(hiddenLayer_size, 3)
         self.fc3 = nn.Linear(3, output_size)
         
     def forward(self, x):
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = F.relu(self.fc3(x))
         #x = torch.sigmoid(self.fc1(x))
         #x = torch.sigmoid(self.fc2(x))
         return x
 
model = AI()
print(model)

loss_func = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.002)

#%%
for name, parameter in model.named_parameters():
    print('name: ', name)
    print(type(parameter))
    print('param.shape: ', parameter.shape)
    print('param.requires_grad: ', parameter.requires_grad)
    print('=========')

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
    
    #sample[3]       = pow(data[idx]['x'], 2)
    #sample[4]       = pow(data[idx]['y'], 2)
    #sample[5]       = pow(data[idx]['z'], 2)
    
    output_ref[0]   = data[idx]['density']
    
    return sample, output_ref

print(sampleFromDataset(training_data, sample, output))


#%%
print('Training model')

num_epochs = 20000

for i in range(num_epochs):
    sampleFromDataset(training_data, sample, output)
    pred = model(torch.autograd.Variable(torch.FloatTensor(sample)))
    loss = loss_func(pred, torch.autograd.Variable(torch.FloatTensor(output)))
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('Finished training model')
#%%
# =============================================================================
# def writeLayer(weights, key, file):
#     if len(list(weights[key].shape)) == 2:
#         for y in range(weights[key].shape[0]):
#             line = ""
#             for x in range(weights[key].shape[1]):
#                 line = line + " " + str(float(weights[key][y][x]))
#             line = line + "\n"
#             file.write(line)
#     else:
#         line = ""
#         for x in range(weights[key].shape[0]):
#             line = line + " " + str(float(weights[key][x]))
#         line = line + "\n"
#         file.write(line)
# =============================================================================

def writeLayer(weights, key, file):
    if len(list(weights[key].shape)) == 2:
        for y in range(weights[key].shape[0]):
            for x in range(weights[key].shape[1]):
                line = str(float(weights[key][y][x])) + "\n"
                file.write(line)
    else:
        for x in range(weights[key].shape[0]):
            line = str(float(weights[key][x])) + "\n"
            file.write(line)

def write(weights, file):
    keys = weights.keys()
    
    for k in keys:
        writeLayer(weights, k, file) 
#%%
# https://towardsdatascience.com/everything-you-need-to-know-about-saving-weights-in-pytorch-572651f3f8de
print('output weights')
weights = model.state_dict()

for key in weights:
    print('key: ', key)
    print('shape: ', weights[key].shape)
        
    if len(list(weights[key].shape)) == 2:
        for y in range(weights[key].shape[0]):
            for x in range(weights[key].shape[1]):
                print(key, "yx: " + str(float(weights[key][y][x])))
    else:
        for x in range(weights[key].shape[0]):
            print(key, "x:" + str(float(weights[key][x])))
    print('=========')
   
#%%    
file = open("D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Weights.txt", "w")
write(weights, file)
file.close()

#%%
# This function writes a HLSL function for calulating a hidden layer. This function is stored inside a cginc file that
# will be included with the main shader
def calculateLayerHLSL(file, input_size, num_nodes, weights_size_1, weights_size_2, bias_size):
    param_1 = "float inputs[" + str(input_size) + "]"
    param_2 = ", inout float node[" + str(num_nodes) + "]"
    param_3 = ", float weights[" + str(weights_size_1) + "][" + str(weights_size_2) + "]"
    param_4 = ", float bias["+ str(bias_size) + "]"

    file.write("void calculate_layer(" + param_1 + param_2 + param_3 + param_4 + ")\n{")
    file.write("\n\tint column = 0;\n")
    file.write("\tfor (int n = 0; n < " + str(num_nodes) + "; n++)\n\t{\n")
    file.write("\t\tfor (int row = 0; row < " + str(weights_size_1) + "; row++)\n\t\t{\n")
    file.write("\t\t\tnode[n] += relu(nn_dot(inputs[row], weights[row][column], bias[row]));\n\t\t}\n\t\tcolumn++;\n")
    file.write("\t}\n")
    file.write("}") 

# This function writes a hlsl function that caclualtes the outputs from a hidden layer
def calculateOutputHLSL(file, input_size, num_nodes, weights_size_1, bias_size):
    param_1 = "float inputs[" + str(input_size) + "]"
    param_2 = ", float weights[" + str(weights_size_1) + "]"
    param_3 = ", float bias["+ str(bias_size) + "]"
    
    file.write("void calculate_output(" + param_1 + param_2 + param_3 + ")\n{")
    file.write("\n\t float output = 0.0f;\n")
    file.write("\t for (int row = 0; row < " + str(weights_size_1) + "; row++)\n\t{\n")
    file.write("\t\t node[n] += sigmoid(nn_dot(inputs[row], weights[row], bias[row]));\n\t}\n \t return output;\n")
    file.write("}")
    
def sampleFromWeightsHLSL(file, num_hidden_layers, hidden_layer_size):
    file.write("float sample_from_weights(float3 position)\n{\n")
    file.write("\t float pos[3] = { position.x, position.y, position.z };\n")
    for i in range(num_hidden_layers):
        file.write("\t float layer_" + str(i + 1) + "_nodes[" + str(hidden_layer_size) + "] = {0.0f, 0.0f, 0.0f};\n")
   
    file.write("calculateLayerHLSL(pos, )")

hlslFile = open("D:\\My Documents\\Github\\Year 3\\Dissertation\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\functions.txt", "w")
calculateLayerHLSL(hlslFile, 3, 3, 3, 3, 3)
#calculateOutputHLSL(hlslFile, 3, 3, 3, 1)
#sampleFromWeightsHLSL(hlslFile, 2, 3)