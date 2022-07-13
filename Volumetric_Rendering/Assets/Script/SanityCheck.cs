using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEngine;
using Random = UnityEngine.Random;

public class SanityCheck : MonoBehaviour
{
    // Public property 
    [SerializeField] public TextAsset weightsTextAsset;
    [SerializeField] public TextAsset biasTextAsset;
    public GameObject observer;
    public Vector2 jitterAmount = new Vector2(-0.5f, 0.5f);
    [Range(0, 1)] public float scale = 0.2f;
    public bool enableDebugText = false;
    public bool enableDebugRay = false;
    public int step = 10;
    
    // Private Property
    
    // NN weights text asset
    private List<float> m_Weights = new List<float>();
    private List<float> m_biases = new List<float>();
    
    // private readonly List<Vector3> m_w = new List<Vector3>();
    // private readonly List<Vector3> m_biases = new List<Vector3>();

    private readonly List<Ray> m_RayPool = new List<Ray>();
    private readonly List<Vector3> m_JitterPool = new List<Vector3>();
    private readonly List<GameObject> m_ObjectPool = new List<GameObject>();
    private readonly List<float> m_DensityPool = new List<float>();

    private Bounds m_Box;
    private static int numPoints = 50;
    
    private void ParseWeights(TextAsset file, ref List<float> list)
    {
        // Read the file and return a string
        string contents = file.text;

        // Regex is used to split the contents of the file into separate lines
        string[] lines = Regex.Split(contents, "\n");

        for (int i = 0; i < lines.Length; i++)
        {
            if (lines[i] == "") break;
            list.Add(float.Parse(lines[i]));
        }

        // m_w.Add(new Vector3(m_Weights[0], m_Weights[1], m_Weights[2]));
        // m_w.Add(new Vector3(m_Weights[3], m_Weights[4], m_Weights[5]));
        // m_w.Add(new Vector3(m_Weights[6], m_Weights[7], m_Weights[8]));
        // m_biases.Add(new Vector3(m_Weights[9], m_Weights[10], m_Weights[11]));
        //
        // m_w.Add(new Vector3(m_Weights[12], m_Weights[13], m_Weights[14]));
        // m_w.Add(new Vector3(m_Weights[15], m_Weights[16], m_Weights[17]));
        // m_w.Add(new Vector3(m_Weights[18], m_Weights[19], m_Weights[20]));
        // m_biases.Add(new Vector3(m_Weights[21], m_Weights[22], m_Weights[23]));
        //
        // m_w.Add(new Vector3(m_Weights[24], m_Weights[25], m_Weights[26]));
        // m_w.Add(new Vector3(m_Weights[27], m_Weights[28], m_Weights[29]));
        // m_w.Add(new Vector3(m_Weights[30], m_Weights[31], m_Weights[32]));
        // m_biases.Add(new Vector3(m_Weights[33], m_Weights[34], m_Weights[35]));
        //
        // m_w.Add(new Vector3(m_Weights[36], m_Weights[37], m_Weights[38]));
        // m_biases.Add(new Vector3(m_Weights[39], m_Weights[39], m_Weights[39]));
    }

    private float Relu(float x)
    {
        return Math.Max(0.0f, x);
    }

    private float Sigmoid(float x)
    {
        return 1.0f / (1.0f + Mathf.Exp(-x));
    }

    Vector3 Divide(Vector3 a, Vector3 b)
    {
        float x = a.x / b.x;
        float y = a.y / b.y;
        float z = a.z / b.z;
        return new Vector3(x, y, z);
    }

    Vector3 Min(Vector3 a, Vector3 b)
    {
        return new Vector3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z);
    }
    
    Vector3 Max(Vector3 a, Vector3 b)
    {
        return new Vector3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
    }
    
    private Vector2 BoxIntersection(Ray ray, Bounds bounds)
    {
        Vector3 t0 = Divide((bounds.min - ray.origin), ray.direction);
        Vector3 t1 = Divide((bounds.max - ray.origin), ray.direction);

        Vector3 tMin = Min(t0, t1);
        Vector3 tMax = Max(t0, t1);

        float distA = Math.Max(Math.Max(tMin.x, tMin.y), tMin.z);
        float distB = Math.Min(tMax.x, Math.Min(tMax.y, tMax.z));

        float distToBox = Math.Max(0, distA);
        float distInsideBox = Math.Max(0, distB - distToBox);

        return new Vector2(distToBox, distInsideBox);
    }
    
    private float Density(Vector3 samplePosition)
    {
        Vector3 localToWorld = transform.TransformPoint(samplePosition);
        
        // Input -> hidden layer 1
        float n1 = Relu(m_Weights[0] * localToWorld.x + m_Weights[1] * localToWorld.y + m_Weights[2] * localToWorld.z + m_biases[0]);
        float n2 = Relu(m_Weights[3] * localToWorld.x + m_Weights[4] * localToWorld.y + m_Weights[5] * localToWorld.z + m_biases[1]);
        float n3 = Relu(m_Weights[6] * localToWorld.x + m_Weights[7] * localToWorld.y + m_Weights[8] * localToWorld.z + m_biases[2]);

        // Hidden layer 1 -> hidden layer 2
        float n4 = Relu(m_Weights[9] * n1 + m_Weights[10] * n2 + m_Weights[11] * n3 + m_biases[3]);
        float n5 = Relu(m_Weights[12] * n1 + m_Weights[13] * n2 + m_Weights[14] * n3 + m_biases[4]);
        float n6 = Relu(m_Weights[15] * n1 + m_Weights[16] * n2 + m_Weights[17] * n3 + m_biases[5]);
										
        // Hidden layer 2 -> hidden layer 3
        float n7 = Relu(m_Weights[18] * n4 + m_Weights[19] * n5 + m_Weights[20] * n6 + m_biases[6]);
        float n8 = Relu(m_Weights[21] * n4 + m_Weights[22] * n5 + m_Weights[23] * n6 + m_biases[7]);
        float n9 = Relu(m_Weights[24] * n4 + m_Weights[25] * n5 + m_Weights[26] * n6 + m_biases[8]);
	
        // Hidden layer 3 -> output
        return Sigmoid(m_Weights[27] * n7 + m_Weights[28] * n8 + m_Weights[29] * n9 + m_biases[9]);
    }
    
    void Start()
    {
        // A bounding box is created around the empty game object and is scaled to a unit cube size.
        m_Box.center = transform.position;
        m_Box.min = new Vector3(-1.0f, -1.0f, -1.0f);
        m_Box.max = new Vector3(1.0f, 1.0f, 1.0f);
        
        ParseWeights(weightsTextAsset, ref m_Weights);
        ParseWeights(biasTextAsset, ref m_biases);
        
        // A set of pools is initialize for updating.
        for (int r = 0; r < numPoints; r++)
        {
            m_RayPool.Add(new Ray());
            m_JitterPool.Add(new Vector3(Random.Range(jitterAmount.x, jitterAmount.y), Random.Range(jitterAmount.x, jitterAmount.y), Random.Range(jitterAmount.x, jitterAmount.y)));
            
            for (int sample = 0; sample < step; sample++)
            {
                GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                cube.transform.position = Vector3.zero;
                cube.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
                m_ObjectPool.Add(cube);
                m_DensityPool.Add(0.0f);
            }
        }
    }

    private void Update()
    {
        for (int p = 0; p < numPoints; p++)
        {
            // A ray is created from the observer with its direction jittered to create a spread of rays.
            Vector3 origin = observer.transform.position;
            Vector3 direction = observer.transform.forward;
            m_RayPool[p] = new Ray(origin, direction + m_JitterPool[p]);
            
            // Each ray is intersected with the bounding box and the distance to the box and the distance inside the box
            // is returned.
            Vector2 rayBoxIntersection = BoxIntersection(m_RayPool[p], m_Box);
            float distanceToBox = rayBoxIntersection.x;
            float distanceInsideBox = rayBoxIntersection.y;
            float stepSize = distanceInsideBox / step;
            float distanceToTravel = 0.0f;
            
            // An entry point is made at the intersection point of the bounding box and samples are made along the ray
            // to sample the density of the bounding box using the NN.
            Vector3 entryPoint = m_RayPool[p].origin + m_RayPool[p].direction * distanceToBox;
            for (int i = 0; i < step; i++)
            {
                // Density is evaluated using the sample position along the ray.
                Vector3 samplePosition = entryPoint + m_RayPool[p].direction * distanceToTravel;

                float density = Density(samplePosition);
                m_DensityPool[i * p] = density;
                
                // The pool of cubes have their position set to stay inline with the rays and each cubes material is changed
                // to reflect the density at that point.
                m_ObjectPool[i * p].transform.localScale = new Vector3(scale, scale, scale);
                m_ObjectPool[i * p].transform.position = samplePosition;
                m_ObjectPool[i * p].GetComponent<Renderer>().material.color = new Color(density, density, density);
                
                distanceToTravel += stepSize;
            }
        }
    }

    private void OnDrawGizmos()
    {
        // Draw each point of the bounding box
        Gizmos.DrawSphere(m_Box.min, 0.1f);
        Gizmos.DrawSphere(m_Box.max, 0.1f);
        Gizmos.DrawWireCube(m_Box.center, m_Box.size);
        
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(Vector3.zero, 0.8f);


        if (enableDebugRay)
        {
            // Draw each ray from the observer
            foreach (var ray in m_RayPool)
            {
                Gizmos.DrawRay(ray.origin, ray.direction * 5.0f);
            }
        }
        
        if (!enableDebugText) return;
        for (int i = 0; i < m_ObjectPool.Count; i++)
        {
            Handles.Label(m_ObjectPool[i].transform.position, m_DensityPool[i].ToString(CultureInfo.CurrentCulture));
        }
    }
}


// // Input -> hidden layer 1
// float n1 = Relu(m_w[0].x * localToWorld.x + m_w[0].y * localToWorld.y + m_w[0].z * localToWorld.z + m_biases[0].x);
// float n2 = Relu(m_w[1].x * localToWorld.x + m_w[1].y * localToWorld.y + m_w[1].z * localToWorld.z + m_biases[0].y);
// float n3 = Relu(m_w[2].x * localToWorld.x + m_w[2].y * localToWorld.y + m_w[2].z * localToWorld.z + m_biases[0].z);
//
// // Hidden layer 1 -> hidden layer 2
// float n4 = Relu(m_w[3].x * n1 + m_w[3].y * n2 + m_w[3].z * n3 + m_biases[1].x);
// float n5 = Relu(m_w[4].x * n1 + m_w[4].y * n2 + m_w[4].z * n3 + m_biases[1].y);
// float n6 = Relu(m_w[5].x * n1 + m_w[5].y * n2 + m_w[5].z * n3 + m_biases[1].z);
// 		
// // Hidden layer 2 -> hidden layer 3
// float n7 = Relu(m_w[6].x * n4 + m_w[6].y * n5 + m_w[6].z * n6 + m_biases[2].x);
// float n8 = Relu(m_w[7].x * n4 + m_w[7].y * n5 + m_w[7].z * n6 + m_biases[2].y);
// float n9 = Relu(m_w[8].x * n4 + m_w[8].y * n5 + m_w[8].z * n6 + m_biases[2].z);
//
// // Hidden layer 3 -> output
// return Sigmoid(m_w[9].x * n7 + m_w[9].y * n8 + m_w[9].z * n9 + m_biases[3].x);