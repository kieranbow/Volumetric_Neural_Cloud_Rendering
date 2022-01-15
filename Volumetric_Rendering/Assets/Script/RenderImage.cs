using System;
using System.Collections;
using System.Collections.Generic;
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
    private void Start()
    {
        // Presuming there is only 1 camera and that is the main camera, set the depth buffers
        if (Camera.main == null) return;
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
        
        if (material == null)
        {
            material = new Material(shader);
        }
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
        
        // Set box bounds to shader
        material.SetVector("_BoundsMin", position - localScale / 2);
        material.SetVector("_BoundsMax", position + localScale / 2);
        material.SetTexture("Shape_tex", shapeTexture3D);
        material.SetTexture("Noise_tex", detailTexture3D);
        material.SetTexture("Weather_tex", weatherTexture2D);
        
        Graphics.Blit(source, destination, material);
    }
}
