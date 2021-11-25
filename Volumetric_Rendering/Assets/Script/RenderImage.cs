using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RenderImage : MonoBehaviour
{
    private Material material;
    [SerializeField] public Shader shader;

    private void Start()
    {
        // Presuming there is only 1 camera and thats the main camera, sample the depth buffers
        Camera.main.depthTextureMode = DepthTextureMode.Depth;
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (material == null)
        {
            material = new Material(shader);
            material.hideFlags = HideFlags.HideAndDontSave; // Don't know what this does
        }

        Matrix4x4 viewToWorld = Camera.main.cameraToWorldMatrix;
        material.SetMatrix("_ViewToWorld", viewToWorld);

        Graphics.Blit(source, destination, material);
    }}
