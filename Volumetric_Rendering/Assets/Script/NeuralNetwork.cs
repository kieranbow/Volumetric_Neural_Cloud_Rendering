using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace NeuralNetwork
{
    public static class NeuralNetworkHelperFunction
    {
        public static void CreateListOfWeights(List<float> weightsFromFile, ref List<Vector4> weightsForShader)
        {
            if (weightsForShader == null) return;
            
            // Hidden Layer 1 weights and biases
            weightsForShader.Add(new Vector4(weightsFromFile[0], weightsFromFile[1], weightsFromFile[2], weightsFromFile[9]));
            weightsForShader.Add(new Vector4(weightsFromFile[3], weightsFromFile[4], weightsFromFile[5], weightsFromFile[10]));
            weightsForShader.Add(new Vector4(weightsFromFile[6], weightsFromFile[7], weightsFromFile[8], weightsFromFile[11]));

            // Hidden Layer 2 weights and biases
            weightsForShader.Add(new Vector4(weightsFromFile[12], weightsFromFile[13], weightsFromFile[14], weightsFromFile[21]));
            weightsForShader.Add(new Vector4(weightsFromFile[15], weightsFromFile[16], weightsFromFile[17], weightsFromFile[22]));
            weightsForShader.Add(new Vector4(weightsFromFile[18], weightsFromFile[19], weightsFromFile[20], weightsFromFile[23]));

            // Hidden Layer 3 weights and biases
            weightsForShader.Add(new Vector4(weightsFromFile[24], weightsFromFile[25], weightsFromFile[26], weightsFromFile[33]));
            weightsForShader.Add(new Vector4(weightsFromFile[27], weightsFromFile[28], weightsFromFile[29], weightsFromFile[34]));
            weightsForShader.Add(new Vector4(weightsFromFile[30], weightsFromFile[31], weightsFromFile[32], weightsFromFile[35]));

            // Output weights and bias
            weightsForShader.Add(new Vector4(weightsFromFile[36], weightsFromFile[37], weightsFromFile[38], weightsFromFile[39]));
        }

        public static void SendWeightsToShader(ref Material material, List<Vector4> weightPacket)
        {
            if (material == null) return;
            
            material.SetVector(Shader.PropertyToID("weight0"), weightPacket[0]);
            material.SetVector(Shader.PropertyToID("weight1"), weightPacket[1]);
            material.SetVector(Shader.PropertyToID("weight2"), weightPacket[2]);
            material.SetVector(Shader.PropertyToID("weight3"), weightPacket[3]);
            material.SetVector(Shader.PropertyToID("weight4"), weightPacket[4]);
            material.SetVector(Shader.PropertyToID("weight5"), weightPacket[5]);
            material.SetVector(Shader.PropertyToID("weight6"), weightPacket[6]);
            material.SetVector(Shader.PropertyToID("weight7"), weightPacket[7]);
            material.SetVector(Shader.PropertyToID("weight8"), weightPacket[8]);
            material.SetVector(Shader.PropertyToID("weight9"), weightPacket[9]);
        }
    }
}