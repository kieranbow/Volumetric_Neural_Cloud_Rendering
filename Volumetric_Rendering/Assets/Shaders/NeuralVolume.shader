Shader "Unlit/NeuralVolume"
{
    Properties
    {
        [HideInInspector] _MainTex ("Texture", 2D) = "white" {}
        _NumStep ("Number of Steps", Range(0, 256)) = 100
        _DensityMulti ("Density Multiplier", Range(0, 1000)) = 1.0
        _Min ("Min debug", Range(0, 1)) = 0
        _Max ("Max debug", Range(0, 1)) = 1
    }
    SubShader
    {
        // No culling or depth
        Cull Off 
        ZWrite Off 
        ZTest Always
        
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            // Includes
            #include "UnityCG.cginc"

            // Custom Includes
            #include "cginc/Ray.cginc"
            #include "cginc/Shapes.cginc"
            #include "cginc/GeneratedFunction.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
                float3 view_vector : TEXCOORD1;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;

                float3 view_vector = mul(unity_CameraInvProjection, float4(v.uv * 2.0f - 1.0f, 0.0f, -1.0f));
                o.view_vector = mul(unity_CameraToWorld, float4(view_vector, 0.0f));
                
                return o;
            }
            
            sampler2D _MainTex, _CameraDepthTexture;
            float4 _MainTex_ST;

            float3 _BoundsMin, _BoundsMax;
            int _NumStep;
            float _DensityMulti, _Min, _Max;

            // float4 weight0;
            // float4 weight1;
            // float4 weight2;
            // float4 weight3;
            // float4 weight4;
            // float4 weight5;
            // float4 weight6;
            // float4 weight7;
            // float4 weight8;
            // float4 weight9;

            float weights[114];
            float bias[19];

            float sampleDensity(float3 position)
            {
                // Convert the world space position into local space
                float3 world_to_local = mul(unity_WorldToObject, float4(position, 1.0f)).xyz;

                // float4 hiddenLayer1Weights[3] = { weight0, weight1, weight2 };
                // float4 hiddenLayer2Weights[3] = { weight3, weight4, weight5 };
                // float4 hiddenLayer3Weights[3] = { weight6, weight7, weight8 };

                // HiddenLayer hidden_layer1 = makeHiddenLayer(weight0, weight1, weight2);
                // HiddenLayer hidden_layer2 = makeHiddenLayer(weight3, weight4, weight5);
                // HiddenLayer hidden_layer3 = makeHiddenLayer(weight6, weight7, weight8);
                // HiddenLayer output = makeHiddenLayer(weight9, weight9, weight9);
                //
                float input[6] = { world_to_local.x, world_to_local.y, world_to_local.z, pow(world_to_local.x, 2), pow(world_to_local.y, 2), pow(world_to_local.z, 2) };
                //
                // calculateLayer(input, hidden_layer1);
                // calculateLayer(hidden_layer1.neurons, hidden_layer2);
                // calculateLayer(hidden_layer2.neurons, hidden_layer3);
                // return outputNetwork(output, hidden_layer3);

                return calculateDensityFromANN(input, weights, bias);
                
                // const float3 hiddenLayer1 = calculate_layer(world_to_local, hiddenLayer1Weights);
                // const float3 hiddenLayer2 = calculate_layer(hiddenLayer1, hiddenLayer2Weights);
                // const float3 hiddenLayer3 = calculate_layer(hiddenLayer2, hiddenLayer3Weights);
                //
                // return outputNetwork(hiddenLayer3, weight9) * _DensityMulti;


                
                // // Input -> hidden layer 1
                // const float n_1 = relu(weight0.x * world_to_local.x + weight0.y * world_to_local.y + weight0.z * world_to_local.z + weight0.w);
                // const float n_2 = relu(weight1.x * world_to_local.x + weight1.y * world_to_local.y + weight1.z * world_to_local.z + weight1.w);
                // const float n_3 = relu(weight2.x * world_to_local.x + weight2.y * world_to_local.y + weight2.z * world_to_local.z + weight2.w);
                //
                // // Hidden layer 1 -> hidden layer 2
                // const float n_4 = relu(weight3.x * n_1 + weight3.y * n_2 + weight3.z * n_3 + weight3.w);
                // const float n_5 = relu(weight4.x * n_1 + weight4.y * n_2 + weight4.z * n_3 + weight4.y);
                // const float n_6 = relu(weight5.x * n_1 + weight5.y * n_2 + weight5.z * n_3 + weight5.w);
                //
                // // Hidden layer 2 -> hidden layer 3
                // const float n_7 = relu(weight6.x * n_4 + weight6.y * n_5 + weight6.z * n_6 + weight6.w);
                // const float n_8 = relu(weight7.x * n_4 + weight7.y * n_5 + weight7.z * n_6 + weight7.w);
                // const float n_9 = relu(weight8.x * n_4 + weight8.y * n_5 + weight8.z * n_6 + weight8.w);
                //
                // // Hidden layer 3 -> output
                // return sigmoid(weight9.x * n_7 + weight9.y * n_8 + weight9.z * n_9 + weight9.w) * _DensityMulti;
            }

            fixed4 frag (const v2f i) : SV_Target
            {
                // Primary camera ray
                Ray primary_ray;
                primary_ray.origin = _WorldSpaceCameraPos;
                primary_ray.direction = normalize(i.view_vector);
                
                // Depth texture
                const float non_linear_depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
                const float z_depth = LinearEyeDepth(non_linear_depth) * length(i.view_vector);
                
                // Box data
                box box;
                box.bound_max = _BoundsMax;
                box.bound_min = _BoundsMin;
                
                // Ray box intersection
                const float2 ray_box = ray_box_dist(box.bound_min, box.bound_max, primary_ray);
                const float dist_to_box = ray_box.x;
                const float dist_inside_box = ray_box.y;

                // Ray entry point
                const float3 entry_point = primary_ray.origin + primary_ray.direction * dist_to_box;

                float dist_travelled = 0.0f;
                const float step_size = dist_inside_box / _NumStep;
                const float dist_limit = min(z_depth - dist_to_box, dist_inside_box);

                float total_density = 0.0f;
                
                float4 col = float4(0.0f, 0.0f, 0.0f, 0.0f);
                
                while (dist_travelled < dist_limit)
                {
                    const float3 sample_position = entry_point + primary_ray.direction * dist_travelled;
                    const float density = sampleDensity(sample_position);// * step_size;

                    //if (density > _Max) col = float4(1.0f, 1.0f, 1.0f, 1.0f);
                    //if (density < _Min) col = float4(1.0f, 0.0f, 0.0f, 1.0f);
                    
                    total_density += density;
                    dist_travelled += step_size;
                }

                // Combining outputs
                const float3 background_color = tex2D(_MainTex, i.uv).rgb;
                return float4(background_color * exp(-total_density) + col.rgb, 1.0f);
            }
            ENDCG
        }
    }
}
