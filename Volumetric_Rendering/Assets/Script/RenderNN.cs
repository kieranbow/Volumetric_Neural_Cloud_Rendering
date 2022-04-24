using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using UnityEngine;
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

    private List<Color> colourWeights = new List<Color>();
    
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
        
        Color[] temp_weights = colourWeights.ToArray();
        
        Texture2D weightTexture2D = new Texture2D(colourWeights.Count / 4, colourWeights.Count / 4, TextureFormat.RGB24, false, true);
        weightTexture2D.SetPixels(temp_weights);

        byte[] bytes = weightTexture2D.EncodeToPNG();
        var path = Application.dataPath + "/../SaveImages/";

        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
        File.WriteAllBytes(path + "Image"+ ".png", bytes);
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
            Color temp = new Color(float.Parse(lines[i]), 0.0f, 0.0f);
            colourWeights.Add(temp);
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

        Color[] temp_weights = colourWeights.ToArray();
        
        Texture2D weightTexture2D = new Texture2D(colourWeights.Count / 4, colourWeights.Count / 4, TextureFormat.R16, false, true);
        weightTexture2D.SetPixels(temp_weights, 0);
        weightTexture2D.Apply();
        
        material.SetTexture("_weightTex", weightTexture2D);
        
        Bounds bounds = collider.bounds;

        // Send box bounds to the shader
        material.SetVector("_BoundsMin", bounds.min);
        material.SetVector("_BoundsMax", bounds.max);

        Graphics.Blit(source, destination, material);
    }
}
