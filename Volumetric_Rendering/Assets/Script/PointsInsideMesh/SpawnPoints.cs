using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class Position
{
    public string x;
    public string y;
    public string z;
    /*public string d;*/
}

[System.Serializable]
public class Positions
{
    public Position[] positions;
}
[ExecuteInEditMode]
public class SpawnPoints : MonoBehaviour
{
    public TextAsset jsonFile;

    void Start()
    {
        // Create a default sphere for representing each point
        GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
        
        // Read the json file with all of the position
        Positions positionsInJson = JsonUtility.FromJson<Positions>(jsonFile.text);

        // Loop through each position spawning a sphere
        foreach (Position position in positionsInJson.positions)
        {
            float x = float.Parse(position.x);
            float y = float.Parse(position.y);
            float z = float.Parse(position.z);

            Vector3 newPosition = new Vector3(x, y, z);
            
            Instantiate(sphere, newPosition, Quaternion.identity);
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
