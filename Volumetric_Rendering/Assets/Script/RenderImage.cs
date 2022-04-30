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
    [SerializeField] public Texture3D shapeTexture3D;
    [SerializeField] public Texture3D detailTexture3D;
    [SerializeField] public Texture2D weatherTexture2D;
    [SerializeField] public Texture2D blueNoiseTexture2D;

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

            // Create a new render texture which will render the clouds at 1/2 of the screen resolution
            RenderTexture rtClouds = RenderTexture.GetTemporary(source.width, source.height, 0, RenderTextureFormat.R8);
            rtClouds.useDynamicScale = true;
            rtClouds.filterMode = FilterMode.Bilinear;
        
            // Using pass 0 inside the volumetric shader. Render the clouds at 1/2 resolution
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
