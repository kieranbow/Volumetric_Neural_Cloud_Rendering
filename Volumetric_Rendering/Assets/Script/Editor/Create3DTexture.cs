using UnityEditor;
using UnityEngine;

// Code provided by Unity - https://docs.unity3d.com/Manual/class-Texture3D.html
public class Create3DTexture : MonoBehaviour
{
    // https://www.youtube.com/watch?v=Aga0TBJkchM
    private static float Perlin3D(float x, float y, float z)
    {
        // Get all permutations of noise for x, y, z
        var ab = Mathf.PerlinNoise(x, y);
        var bc = Mathf.PerlinNoise(y, z);
        var ac = Mathf.PerlinNoise(x, z);
        
        // Get all reversed permutations of noise for x, y, z
        var ba = Mathf.PerlinNoise(y, x);
        var cb = Mathf.PerlinNoise(x, y);
        var ca = Mathf.PerlinNoise(z, x);

        // Return the average
        return (ab + bc + ac + ba + cb + ca) / 6.0f;
    }
    
    [MenuItem("Textures/3DTexture")]
    private static void CreateTexture3D()
    {
        // Configure the texture
        int size = 32;
        TextureFormat format = TextureFormat.RGBA32;
        TextureWrapMode wrapMode = TextureWrapMode.Clamp;

        // Create the texture and apply the configuration
        Texture3D texture = new Texture3D(size, size, size, format, false);
        texture.wrapMode = wrapMode;

        // Create a 3-dimensional array to store color data
        Color[] colors = new Color[size * size * size];

        // Populate the array so that the x, y, and z values of the texture will map to red, blue, and green colors
        float inverseResolution = 1.0f / (size - 1.0f);
        for (int z = 0; z < size; z++)
        {
            int zOffset = z * size * size;
            for (int y = 0; y < size; y++)
            {
                int yOffset = y * size;
                for (int x = 0; x < size; x++)
                {
                    if (Perlin3D(x * 0.9f, y * 0.9f, z * 0.9f) >= 0.5f)
                    {
                        colors[x + yOffset + zOffset] = new Color(x * inverseResolution, y * inverseResolution, z * inverseResolution, 1.0f);
                    }
                }
            }
        }

        // Copy the color values to the texture
        // texture.SetPixels(colors);

        // Apply the changes to the texture and upload the updated texture to the GPU
        texture.Apply();

        // Save the texture to your Unity Project
        AssetDatabase.CreateAsset(texture, "Assets/Assets/Textures/Example3DTexture.asset");
    }
}
