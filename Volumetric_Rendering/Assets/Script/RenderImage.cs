using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RenderImage : MonoBehaviour
{
    [SerializeField] private Material material;

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Matrix4x4 viewToWorld = Camera.main.cameraToWorldMatrix;
        material.SetMatrix("_ViewToWorld", viewToWorld);
        Graphics.Blit(source, destination, material);
    }}
