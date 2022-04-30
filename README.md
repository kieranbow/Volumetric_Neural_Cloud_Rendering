# Volumetric_Neural_Cloud_Rendering
A repository for my dissertation on Volumetric Neural Cloud Rendering in Unity

## Introduction
This project looks into the usage of a neural network that can produce and render volumetric densities similar to clouds using shaders developed in Unity. A pipeline is created which produces the volumetric data and trains a neural network on. The weights and biases from this network are then used to render the volumetric data in Unity.

## Project Features
This repositiry contains 4 different projects that make up a pipeline for rendering a neural network in a shader.
These projects are:

### [PointsInsideUnitCube](https://github.com/kieranbow/Volumetric_Neural_Cloud_Rendering/tree/main/PointsInsideUnitCube)
This project which is written in C++ generates a random set of points from within a mesh. These points are saved inside a CSV file which is then exported to the Python project. Loading the mesh is done using Assimp. 

### [Neural Network](https://github.com/kieranbow/Volumetric_Neural_Cloud_Rendering/tree/main/Neural%20Network)
This project contains a single python file that uses PyTorch to create and train a neural network on the dataset from PointsInsideUnitCube. The weights and biases from the network are outputted into a text file which is then used in Unity.

### [ShaderSanityCheck](https://github.com/kieranbow/Volumetric_Neural_Cloud_Rendering/tree/main/ShaderSanityCheck)
In order to check if what the neural network has learned is working. This project which is written in C++, replicate the shader that will be used for rendering which can then be debugged to see if the values are correct.

### [Volumetric_Rendering](https://github.com/kieranbow/Volumetric_Neural_Cloud_Rendering/tree/main/Volumetric_Rendering)
This is the heart of the whole project. It's a Unity project which has the neural network shader and the volumetric shader which are used for comparison in a report. The volumetric shader tries to replicate common volumetric cloud implimentation seen both online and in video games. The neural network shader replicates the architecture of the network and uses the weigths and biases to try and render volumetric densities.

## Instructions for using projects
- The PointsInsideUnitCube project and ShaderSanityCheck uses Visual Studio 2019.
- The Volumetric_Rendering project uses Unity 2021.1.21f1.
- The Neural Network project was developed using Anaconda and Spyder so it's recommended to use the latest version of those software. 
