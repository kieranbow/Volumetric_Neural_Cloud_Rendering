using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Serialization;

[ExecuteInEditMode] // Allows the script to be executed in editor
[ImageEffectAllowedInSceneView] // Allows image effects to appear in editor
public class RenderImage : MonoBehaviour
{
    [SerializeField] public Material material;
    [SerializeField] public Shader shader;
    [SerializeField] public Transform box;
    [FormerlySerializedAs("volumeTexture3D")] [SerializeField] public Texture3D shapeTexture3D;
    [SerializeField] public Texture3D detailTexture3D;
    [SerializeField] public Texture2D weatherTexture2D;
    [SerializeField] public TextAsset weights;

    private Vector3 m_Fc1Weights1;
    private Vector3 m_Fc1Weights2;
    private Vector3 m_Fc1Weights3;
    private Vector3 m_Fc1Bias;
    private Vector3 m_Fc2Weights;
    private float m_Fc2Bias;

    private void Start()
    {
        // Presuming there is only 1 camera and that is the main camera, set the depth buffers
        if (Camera.main == null) return;
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
        
        if (material == null)
        {
            material = new Material(shader);
        }
        
        ParseWeights();

        // Send neural network weights to the shader
        material.SetVector("fc1_weights_1", m_Fc1Weights1);
        material.SetVector("fc1_weights_2", m_Fc1Weights2);
        material.SetVector("fc1_weights_3", m_Fc1Weights3);
        material.SetVector("fc1_bias", m_Fc1Bias);
        material.SetVector("fc2_weights", m_Fc2Weights);
        material.SetFloat("fc2_bias", m_Fc2Bias);
    }

    private void ParseWeights()
    {
        // Read the file and return a string
        string file = weights.text;
        
        string[] lines = Regex.Split(file, "\n");

        m_Fc1Weights1.x = float.Parse(lines[0]);
        m_Fc1Weights1.y = float.Parse(lines[1]);
        m_Fc1Weights1.z = float.Parse(lines[2]);

        m_Fc1Weights2.x = float.Parse(lines[3]);
        m_Fc1Weights2.y = float.Parse(lines[4]);
        m_Fc1Weights2.z = float.Parse(lines[5]);

        m_Fc1Weights3.x = float.Parse(lines[6]);
        m_Fc1Weights3.y = float.Parse(lines[7]);
        m_Fc1Weights3.z = float.Parse(lines[8]);

        m_Fc1Bias.x = float.Parse(lines[9]);
        m_Fc1Bias.y = float.Parse(lines[10]);
        m_Fc1Bias.z = float.Parse(lines[11]);

        m_Fc2Weights.x = float.Parse(lines[12]);
        m_Fc2Weights.y = float.Parse(lines[13]);
        m_Fc2Weights.z = float.Parse(lines[14]);

        m_Fc2Bias = float.Parse(lines[15]);

        // for (int i = 0; i < lines.Length; i++)
        // {
        //     Debug.Log("line: " + float.Parse(lines[i]));
        // }
    }
    
    // https://docs.unity3d.com/ScriptReference/MonoBehaviour.OnRenderImage.html
    // http://www.benmandrew.com/articles/custom-post-processing-effects-in-unity
    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (material == null)
        {
            material = new Material(shader);
        }
        
        // Get boxes position and scale
        var position = box.position;
        var localScale = box.localScale;
        
        // Send box bounds to the shader
        material.SetVector("_BoundsMin", position - localScale / 2);
        material.SetVector("_BoundsMax", position + localScale / 2);
        
        // Send textures to the shader
        material.SetTexture("Shape_tex", shapeTexture3D);
        material.SetTexture("Noise_tex", detailTexture3D);
        material.SetTexture("Weather_tex", weatherTexture2D);
        
        Graphics.Blit(source, destination, material);
    }
}
