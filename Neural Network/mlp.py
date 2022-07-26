#%% Import libraries
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import random
import math

#%% CSV Dataset class
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


training_data   = CSVdataset('F:\\Unity\\Projects\\Volumetric_Neural_Cloud_Rendering\\PointsInsideUnitCube\\PointsInsideUnitCube\\Training.csv')

# for i in range(len(training_data)):
#      sample = training_data[i]
#      print(i, sample['x'], sample['y'], sample['z'], sample['density'])

#%% Neural Network Parameters
inputSize = 9
hiddenLayerSize = 9
biasSize = 9
numLayer = 3
outputSize = 1

#%% Neural Network Class
class AI(nn.Module):
    def __init__(self, input_size, output_size, hiddenLayer_size):
        super().__init__()
         
        self.fc1 = nn.Linear(input_size, hiddenLayer_size)
        self.fc2 = nn.Linear(hiddenLayer_size, hiddenLayer_size)
        self.fc3 = nn.Linear(hiddenLayer_size, hiddenLayer_size)
        self.fc4 = nn.Linear(hiddenLayer_size, output_size)
         
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
 
# Create instant of the AI class and print the topology of the network
model = AI(inputSize, outputSize, hiddenLayerSize)
print(model)

loss_func = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.03)

#%% Output the parameters of the model
for name, parameter in model.named_parameters():
    print('name: ', name)
    print(type(parameter))
    print('param.shape: ', parameter.shape)
    print('param.requires_grad: ', parameter.requires_grad)
    print('=========')

#%% Sampling randomly from the dataset
sample = np.array([0,0,0,0,0,0,0,0,0], dtype=float)
output = np.array([0], dtype=float)

# Randomly sample a line in the dataset and extract x, y, z, density 
# and return it
def sampleFromDataset(data, sample, output_ref):
    idx = random.randrange(len(data))
    sample[0]       = data[idx]['x']
    sample[1]       = data[idx]['y']
    sample[2]       = data[idx]['z']
    sample[3]       = pow(data[idx]['x'], 2)
    sample[4]       = pow(data[idx]['y'], 2) 
    sample[5]       = pow(data[idx]['z'], 2) 
    sample[6]       = math.sin(data[idx]['x'] * 4 * math.pi)
    sample[7]       = math.sin(data[idx]['y'] * 4 * math.pi)
    sample[8]       = math.sin(data[idx]['z'] * 4 * math.pi)
    output_ref[0]   = data[idx]['density'] 
    return sample, output_ref

print(sampleFromDataset(training_data, sample, output))

#%% Train the model
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
torch.save(model.state_dict(), "F:\\Unity\\Projects\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Models\\MLP_model.pt")

#%%
model2 = AI() 
model2.load_state_dict(torch.load("D:\\My Documents\\Github\\Year 3\\Dissertation\Volumetric_Neural_Cloud_Rendering\\Neural Network\\MLP_model.pt"))

#%%
sample_test = np.array([0,0,0], dtype=float)
sampleFromDataset(training_data, sample_test, output)

pred_1 = model(torch.autograd.Variable(torch.FloatTensor(sample_test)))
print(torch.log(pred_1))

pred_2 = model2(torch.autograd.Variable(torch.FloatTensor(sample_test)))
print(torch.log(pred_1))

#%% 
def writeLayer(weights, key, file, file2):
    if len(list(weights[key].shape)) == 2:
        for y in range(weights[key].shape[0]):
            for x in range(weights[key].shape[1]):
                line = str(float(weights[key][y][x])) + "\n"
                file.write(line)
    else:
        for x in range(weights[key].shape[0]):
            line = str(float(weights[key][x])) + "\n"
            file2.write(line)

# Output the contents of the weight from the neural network
def write(weights, file, file2):
    keys = weights.keys()
    
    for k in keys:
        writeLayer(weights, k, file, file2) 

#%% Output Weights and bias
# https://towardsdatascience.com/everything-you-need-to-know-about-saving-weights-in-pytorch-572651f3f8de
print('output weights')
weights = model.state_dict()

for key in weights:
    print('key: ', key)
    print('shape: ', weights[key].shape)

    if len(list(weights[key].shape)) == 2:
        for y in range(weights[key].shape[0]):
            for x in range(weights[key].shape[1]):
                print(key, str(float(weights[key][y][x])))
    else:
        for x in range(weights[key].shape[0]):
            print(key, str(float(weights[key][x])))
    print('=========')
   
#%%    
file = open("F:\\Unity\\Projects\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Weights.txt", "w")
file2 = open("F:\\Unity\\Projects\\Volumetric_Neural_Cloud_Rendering\\Neural Network\\Bias.txt", "w")
write(weights, file, file2)
file.close()
file2.close()

#%%
# This function writes a HLSL function for calulating a hidden layer. This function is stored inside a cginc file that
# will be included with the main shader
def writeHLSL(file):
    file.write("float relu(const float x)\n{ \n\treturn max(0.0f, x);\n}")
    file.write("\nfloat sigmoid(const float x)\n{ \n\treturn 1.0f / (1.0f + exp(-x));\n}")

    neuronIter = 0
    weightsIter = 0
    biasIter = 0
    neuronList = []
    
    file.write("\nfloat calculateDensityFromANN(float input["+ str(inputSize) +"], float weights["+ str(inputSize * hiddenLayerSize * numLayer + inputSize) +"], float bias["+ str(biasSize * numLayer + 1) +"])\n{")
    
    for n in range(0, hiddenLayerSize * numLayer):
        neuronList.append("n" + str(n + 1))

    # Write each hiddenLayer
    for x in range(0, numLayer):
    #{ 
        for y in range(0, hiddenLayerSize):
        #{
            dotProdStr = "\n\tconst float "+ neuronList[neuronIter] +" = relu("

            if neuronIter >= hiddenLayerSize:
            #{
                # Write each neuron in hidden layer
                for z in range(0, hiddenLayerSize):
                #{
                    if z == (hiddenLayerSize - 1):
                        dotProdStr += "weights["+ str(weightsIter) +"] * "+ neuronList[z] +" + bias["+ str(biasIter) +"]);"
                        biasIter += 1
                        weightsIter += 1
                    else:
                        dotProdStr += "weights["+ str(weightsIter) +"] * "+ neuronList[z] +" + "
                        weightsIter += 1
                #}
                neuronIter += 1
                file.write(dotProdStr)
            #}
            else:
            #{
                # Write each neuron in hidden layer
                for z in range(0, hiddenLayerSize):
                #{
                    if z == (hiddenLayerSize - 1):
                        dotProdStr += "weights["+ str(weightsIter) +"] * input["+ str(z) +"] + bias["+ str(biasIter) +"]);"
                        biasIter += 1
                        weightsIter += 1
                    else:
                        dotProdStr += "weights["+ str(weightsIter) +"] * input["+ str(z) +"] + "
                        weightsIter += 1
                #}
                neuronIter += 1
                file.write(dotProdStr)    
            #}     
        #}
        file.write("\n")
    #}
    file.write("\n\treturn sigmoid(weights["+ str(weightsIter) + "] * n7 + weights["+ str(weightsIter + 1) + "] * n8 + weights["+ str(weightsIter + 1) + "] * n9 + bias["+ str(biasIter) + "]);\n}")

hlslFile = open("F:\\Unity\\Projects\\Volumetric_Neural_Cloud_Rendering\\Volumetric_Rendering\\Assets\\Shaders\\cginc\\GeneratedFunction.cginc", 'w')
writeHLSL(hlslFile)

