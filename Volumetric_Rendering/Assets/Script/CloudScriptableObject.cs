using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "Cloud data", menuName = "ScriptableObject/CloudScriptableObject", order = 1)]
public class CloudScriptableObject : ScriptableObject
{
    [field: Header("Textures")] 
    public Texture3D shapeTexture3D;
    public Texture3D detailTexture3D;
    public Texture2D weatherTexture2D;
    public Texture2D blueNoiseTexture2D;

    [field: Header("Neural Network weights")]
    public TextAsset weights;
    
    [Header("Base Cloud Shape")]
    [Range(0.0f, 100.0f)] public float cloudScale = 10.0f;
    public Vector4 cloudOffset;

    [Header("Detailed Cloud Shape")] 
    [Range(0.0f, 500.0f)] public float detailScale = 50.0f;
    public Vector4 detailOffset;

    [Header("Cloud Density")] 
    [Range(0.0f, 10.0f)] public float cloudDensity = 5.0f;

    [Header("Light scattering")]
    [Range(0.0f, 1.0f)] public float inScattering;
    [Range(0.0f, 1.0f)] public float inScatteringIntensity = 1.0f;
    [Range(0.0f, 20.0f)] public float inScatteringExp;
    [Range(0.0f, 1.0f)] public float outScattering = 1.0f;
    [Range(0.0f, 1.0f)] public float outScatteringAO = 0.5f;
    [Range(0.0f, 1.0f)] public float inOutScatteringBias = 0.3f;

    [Header("Cloud Appearance")] 
    [Range(0.0f, 1.0f)] public float coverage = 0.5f;
    [Range(0.0f, 1.0f)] public float attenuation;

    [Header("Weather Map")] 
    public float weatherMapScale = 500.0f;

    [Header("Ray-marching")] 
    [Range(0, 256)] public int numSteps = 100;

}
