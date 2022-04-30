using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEditor;
using UnityEngine.Rendering;

[ExecuteInEditMode] // Allows the script to be executed in editor
[ImageEffectAllowedInSceneView] // Allows image effects to appear in editor
public class RenderNN : MonoBehaviour
{
    [Header("Rendering")]
    [SerializeField] public Material material;
    [SerializeField] public Shader shader;

    [Header("Volumetric Container")]
    [SerializeField] public Collider collider;

    [Header("Neural Network weights")]
    [SerializeField] public TextAsset weights;

    [Header("Weights texture")] 
    public int width = 1;
    public int height = 1;
    
    private List<Color> colourWeights = new List<Color>();
    private List<float> m_Weights = new List<float>();

    private Vector3 fc1_weight_row_1;
    private Vector3 fc1_weight_row_2;
    private Vector3 fc1_weight_row_3;
    private Vector3 fc1_bias;
    
    private Vector3 fc2_weight_row_1;
    private Vector3 fc2_weight_row_2;
    private Vector3 fc2_weight_row_3;
    private Vector3 fc2_bias;
    
    private Vector3 fc3_weight_row_1;
    private Vector3 fc3_weight_row_2;
    private Vector3 fc3_weight_row_3;
    private Vector3 fc3_bias;
    
    private Vector3 fc4_weight_row_1;
    private float fc4_bias;
    
    void Start()
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
        
        // Color[] temp_weights = colourWeights.ToArray();
        //
        // Texture2D weightTexture2D = new Texture2D(width, height, TextureFormat.R16, false, true);
        // weightTexture2D.SetPixels(temp_weights, 0);
        // weightTexture2D.Apply();
        // weightTexture2D.filterMode = FilterMode.Point;
        //
        // material.SetTexture("_weightTex", weightTexture2D);
        //material.SetFloatArray("_weights", m_Weights);

        fc1_weight_row_1 = new Vector3(m_Weights[0], m_Weights[1], m_Weights[2]);
        fc1_weight_row_2 = new Vector3(m_Weights[3], m_Weights[4], m_Weights[5]);
        fc1_weight_row_3 = new Vector3(m_Weights[6], m_Weights[7], m_Weights[8]);
        fc1_bias = new Vector3(m_Weights[9], m_Weights[10], m_Weights[11]);

        fc2_weight_row_1 = new Vector3(m_Weights[12], m_Weights[13], m_Weights[14]);
        fc2_weight_row_2 = new Vector3(m_Weights[15], m_Weights[16], m_Weights[17]);
        fc2_weight_row_3 = new Vector3(m_Weights[18], m_Weights[19], m_Weights[20]);
        fc2_bias = new Vector3(m_Weights[21], m_Weights[22], m_Weights[23]);
        
        fc3_weight_row_1 = new Vector3(m_Weights[24], m_Weights[25], m_Weights[26]);
        fc3_weight_row_2 = new Vector3(m_Weights[27], m_Weights[28], m_Weights[29]);
        fc3_weight_row_3 = new Vector3(m_Weights[30], m_Weights[31], m_Weights[32]);
        fc3_bias = new Vector3(m_Weights[33], m_Weights[34], m_Weights[35]);

        fc4_weight_row_1 = new Vector3(m_Weights[36], m_Weights[37], m_Weights[38]);
        fc4_bias = m_Weights[39];

        // Save the texture to your Unity Project
        //AssetDatabase.CreateAsset(weightTexture2D, "Assets/Assets/Textures/weightsTexture.asset");

        // byte[] bytes = weightTexture2D.EncodeToPNG();
        // var path = Application.dataPath + "/../SaveImages/";
        //
        // if (!Directory.Exists(path))
        // {
        //     Directory.CreateDirectory(path);
        // }
        //File.WriteAllBytes(path + "Image"+ ".png", bytes);
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
            m_Weights.Add(float.Parse(lines[i]));
        }
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

        // This should only be sent to the shader once at the start of the scene
        // However for testing, it helps if the data is sent every times the OnRenderImage 
        // is called.
        material.SetVector("fc1_weights_1", fc1_weight_row_1);
        material.SetVector("fc1_weights_2", fc1_weight_row_2);
        material.SetVector("fc1_weights_3", fc1_weight_row_3);
        material.SetVector("fc1_bias", fc1_bias);
        
        material.SetVector("fc2_weights_1", fc2_weight_row_1);
        material.SetVector("fc2_weights_2", fc2_weight_row_2);
        material.SetVector("fc2_weights_3", fc2_weight_row_3);
        material.SetVector("fc2_bias", fc2_bias);
        
        material.SetVector("fc3_weights_1", fc3_weight_row_1);
        material.SetVector("fc3_weights_2", fc3_weight_row_2);
        material.SetVector("fc3_weights_3", fc3_weight_row_3);
        material.SetVector("fc3_bias", fc3_bias);
        
        material.SetVector("fc4_weights_1", fc4_weight_row_1);
        material.SetFloat("fc4_bias", fc4_bias);

        Graphics.Blit(source, destination, material);
    }
}
