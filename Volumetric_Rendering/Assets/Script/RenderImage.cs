using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[ImageEffectAllowedInSceneView] public class RenderImage : MonoBehaviour
{
    private Material m_Material;
    private Camera m_Camera;
    [SerializeField] public Shader shader;
    [SerializeField] public Transform box;

    private void Start()
    {
        // Presuming there is only 1 camera and that is the main camera, sample the depth buffers
        if (Camera.main == null) return;
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
        
        if (m_Material == null)
        {
            m_Material = new Material(shader);
        }
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (m_Material == null)
        {
            m_Material = new Material(shader);
        }
        
        // Get boxes position and scale
        var position = box.position;
        var localScale = box.localScale;
        
        // Set box bounds to shader
        m_Material.SetVector("_BoundsMin", position - localScale / 2);
        m_Material.SetVector("_BoundsMax", position + localScale / 2);

        // Set noise to shader

        Graphics.Blit(source, destination, m_Material);
    }}
