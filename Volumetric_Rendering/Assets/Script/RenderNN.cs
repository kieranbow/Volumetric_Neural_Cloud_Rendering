using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using NeuralNetwork;

[ExecuteInEditMode] // Allows the script to be executed in editor
[ImageEffectAllowedInSceneView] // Allows image effects to appear in editor
public class RenderNN : MonoBehaviour
{
    [Header("Rendering")] 
    [SerializeField] public Material material;
    [SerializeField] public Shader shader;

    [Header("Neural Network Weights & Bias")] 
    [SerializeField] public TextAsset weightsTextAsset;
    [SerializeField] public TextAsset biasTextAsset;

    [Header("Bounding Box")] [SerializeField]
    public Vector3 boundingBoxcenter = Vector3.zero;
    public Vector3 boundingBoxMin = new Vector3(-1.0f, -1.0f, -1.0f);
    public Vector3 boundingBoxMax = new Vector3(1.0f, 1.0f, 1.0f);
    private Bounds m_Box;

    //private List<float> m_Weights = new List<float>();
    //private List<Vector4> m_weightsPacket = new List<Vector4>();
    private List<float> m_WeightsList = new List<float>();
    private List<float> m_BiasList = new List<float>();

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

        ParseWeights(weightsTextAsset, ref m_WeightsList);
        ParseWeights(biasTextAsset, ref m_BiasList);
        //NeuralNetworkHelperFunction.CreateListOfWeights(m_Weights, ref m_weightsPacket);
        //NeuralNetworkHelperFunction.SendWeightsToShader(ref material, m_weightsPacket);

        material.SetFloatArray(Shader.PropertyToID("weights"), m_WeightsList);
        material.SetFloatArray(Shader.PropertyToID("bias"), m_BiasList);
        
        m_Box.center = boundingBoxcenter;
        m_Box.min = boundingBoxMin;
        m_Box.max = boundingBoxMax;
    }

    // Parse a txt file which contains a list of weights and biases for each layer of a neural network
    private void ParseWeights(TextAsset file, ref List<float> list)
    {
        // Read the file and return a string
        string contents = file.text;

        // Regex is used to split the contents of the file into separate lines
        string[] lines = Regex.Split(contents, "\n");

        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i] == "") break;
            list.Add(float.Parse(lines[i]));
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

        m_Box.center = boundingBoxcenter;
        m_Box.min = boundingBoxMin;
        m_Box.max = boundingBoxMax;

        // Send box bounds to the shader
        
        material.SetVector(Shader.PropertyToID("_BoundsMin"), m_Box.min);
        material.SetVector(Shader.PropertyToID("_BoundsMax"), m_Box.max);

        Graphics.Blit(source, destination, material);
    }

    private void OnDrawGizmos()
    {
        Gizmos.DrawWireCube(m_Box.center, m_Box.size);
        Gizmos.DrawSphere(m_Box.min, 0.1f);
        Gizmos.DrawSphere(m_Box.max, 0.1f);
        //Gizmos.color = Color.red;
        //Gizmos.DrawWireSphere(Vector3.zero, 0.8f);
    }
}
