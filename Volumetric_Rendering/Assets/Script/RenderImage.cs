using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
// [ImageEffectAllowedInSceneView] 
[ExecuteInEditMode]
[ImageEffectAllowedInSceneView]
public class RenderImage : MonoBehaviour
{
    [SerializeField] public Material material;
    [SerializeField] public Shader shader;
    [SerializeField] public Transform box;
    [SerializeField] public Texture3D volumeTexture3D;
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
        material.SetTexture("volume_texture_3d", volumeTexture3D);
 
        // Set noise to shader
        
        Graphics.Blit(source, destination, material);
        
    }
}
