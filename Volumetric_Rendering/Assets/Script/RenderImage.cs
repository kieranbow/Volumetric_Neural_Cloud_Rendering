using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Serialization;

[ExecuteInEditMode] // Allows the script to be executed in editor
[ImageEffectAllowedInSceneView] // Allows image effects to appear in editor
public class RenderImage : MonoBehaviour
{
    [Header("Rendering")]
    [SerializeField] public Material material;
    [SerializeField] public Shader shader;
    public bool enableTwoPassRendering = true;
    
    [Header("Volumetric Container")]
    [SerializeField] public Collider collider;
    
    [Header("Textures")]
    [FormerlySerializedAs("volumeTexture3D")] [SerializeField] public Texture3D shapeTexture3D;
    [SerializeField] public Texture3D detailTexture3D;
    [SerializeField] public Texture2D weatherTexture2D;
    [SerializeField] public Texture2D blueNoiseTexture2D;
    
    [Header("Neural Network weights")]
    [SerializeField] public TextAsset weights;

    private Vector3 m_Fc1Weights1;
    private Vector3 m_Fc1Weights2;
    private Vector3 m_Fc1Weights3;
    private Vector3 m_Fc1Bias;
    private Vector3 m_Fc2Weights;
    private float m_Fc2Bias;

    //[SerializeField] private List<float> m_Weights = new List<float>();

    private List<Color> colourWeights = new List<Color>();

    private void Start()
    {
        // Presuming there is only 1 camera and that is the main camera, set the DepthTextureMode to depth
        if (Camera.main == null) return;
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
        
        if (material == null)
        {
            material = new Material(shader)
            {
                hideFlags = HideFlags.HideAndDontSave
            };
        }

        ParseWeights();
        // Send neural network weights to the shader

        Texture2D weightTexture2D = new Texture2D(colourWeights.Count, 1, TextureFormat.R16, false, true);
        weightTexture2D.SetPixels(colourWeights.ToArray());
        
        material.SetTexture("_weightTex", weightTexture2D);

        // int stride = System.Runtime.InteropServices.Marshal.SizeOf(typeof(float));
        // ComputeBuffer weightsBuffer = new ComputeBuffer(m_Weights.Count, stride, ComputeBufferType.Default);
        // weightsBuffer.SetData(m_Weights.ToArray());
        // material.SetBuffer("weights", weightsBuffer);



        // material.SetVector("fc1_weights_1", m_Fc1Weights1);
        // material.SetVector("fc1_weights_2", m_Fc1Weights2);
        // material.SetVector("fc1_weights_3", m_Fc1Weights3);
        // material.SetVector("fc1_bias", m_Fc1Bias);
        // material.SetVector("fc2_weights", m_Fc2Weights);
        // material.SetFloat("fc2_bias", m_Fc2Bias);
    }

    // Parse a txt file which contains a list of weights and biases for each
    // layer of a neural network
    private void ParseWeights()
    {
        // Read the file and return a string
        string file = weights.text;
        
        // Regex is used to split the contents of the file into separate lines
        string[] lines = Regex.Split(file, "\n");
        
        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i] == "") break;
            // m_Weights.Add(float.Parse(lines[i]));
            
            Color temp = new Color(float.Parse(lines[i]), 0.0f, 0.0f);
            colourWeights.Add(temp);
        }

        // // Weights for input to hidden layer 1
        // m_Fc1Weights1.x = float.Parse(lines[0]);
        // m_Fc1Weights1.y = float.Parse(lines[1]);
        // m_Fc1Weights1.z = float.Parse(lines[2]);
        // m_Fc1Weights2.x = float.Parse(lines[3]);
        // m_Fc1Weights2.y = float.Parse(lines[4]);
        // m_Fc1Weights2.z = float.Parse(lines[5]);
        // m_Fc1Weights3.x = float.Parse(lines[6]);
        // m_Fc1Weights3.y = float.Parse(lines[7]);
        // m_Fc1Weights3.z = float.Parse(lines[8]);
        //
        // // Biases for input to hidden layer 1
        // m_Fc1Bias.x = float.Parse(lines[9]);
        // m_Fc1Bias.y = float.Parse(lines[10]);
        // m_Fc1Bias.z = float.Parse(lines[11]);
        //
        // // Weights for hidden layer 1 to output
        // m_Fc2Weights.x = float.Parse(lines[12]);
        // m_Fc2Weights.y = float.Parse(lines[13]);
        // m_Fc2Weights.z = float.Parse(lines[14]);
        //
        // // Biases for hidden layer 1 to output
        // m_Fc2Bias = float.Parse(lines[15]);
    }
    
    // https://docs.unity3d.com/ScriptReference/MonoBehaviour.OnRenderImage.html
    // http://www.benmandrew.com/articles/custom-post-processing-effects-in-unity
    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (material == null)
        {
            material = new Material(shader)
            {
                hideFlags = HideFlags.HideAndDontSave
            };
        }
        
        Bounds bounds = collider.bounds;

        // Send box bounds to the shader
        material.SetVector("_BoundsMin", bounds.min);
        material.SetVector("_BoundsMax", bounds.max);

        if (enableTwoPassRendering)
        {
            // Send textures to the shader
            material.SetTexture("Shape_tex", shapeTexture3D);
            material.SetTexture("Noise_tex", detailTexture3D);
            material.SetTexture("Weather_tex", weatherTexture2D);
            material.SetTexture("blueNoise_tex", blueNoiseTexture2D);

            // Create a new render texture which will render the clouds at 1/4 of the screen resolution
            RenderTexture rtClouds = RenderTexture.GetTemporary(source.width / 2, source.height / 2, 0, RenderTextureFormat.R8);
            rtClouds.useDynamicScale = true;
            rtClouds.filterMode = FilterMode.Bilinear;
        
            // Using pass 0 inside the volumetric shader. Render the clouds at 1/4 resolution
            Graphics.Blit(source, rtClouds, material, 0);

            // Once the cloud rendering has finished, the next part is to send the render texture back to 
            // the shader. This will then be combined using the shaders pass 1
            material.SetTexture("_cloudsTex", rtClouds);
            Graphics.Blit(source, destination, material, 1);
        
            // Release the renderTexture since it does not need to be used anymore. 
            RenderTexture.ReleaseTemporary(rtClouds); 
        }
        else
        {
            Graphics.Blit(source, destination, material);
        }
    }
}
