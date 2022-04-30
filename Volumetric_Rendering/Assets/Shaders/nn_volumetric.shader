Shader "Custom/nn_volumetric"
{
    Properties
    {
        [HideInInspector] _MainTex ("Texture", 2D) = "white" {}
        
        [Header(Base Shape)]
        [Space(10)]
        _CloudScale ("Cloud Scale", Range(1.0, 100.0)) = 10.0
        _CloudOffset ("Cloud Offset", Vector) = (0,0,0,0)
        
        [Header(Shape Detail)]
        [Space(10)]
        _DetailScale ("Detail Scale", Range(0, 500)) = 50.0
        _DetailOffset("Detail offset", Vector) = (0,0,0,0)
        
        [Header(Cloud Density)]
        [Space(10)]
        _DensityThreshold ("Density Threshold", Range(0, 1)) = 0.5
        _DensityMulti ("Density Multiplier", Range(0, 10)) = 1.0
        
        [Header(Cloud appearance)]
        [Space(10)]
        _globalCoverage ("Cloud coverage", Range(0, 1)) = 0.5
        _in_scattering ("In Scattering", Range(0, 1)) = 0.0
        _out_scattering ("Out Scattering", Range(0, 1)) = 0.0
        _in_out_scattering ("In Out Scattering bias", Range(0, 1)) = 0.0
        _silver_line_intensity ("Silver Lining Intensity", Range(0, 1)) = 0.0
        _silver_line_exp ("Silver Lining Exponent", Range(0, 1)) = 0.0
        _cloud_beer ("Beer amount", Range(0, 1)) = 0.0
        _density_to_sun ("Density to sun", Range(0, 10)) = 0.0
        _cloud_attenuation ("Attenuation", Range(0, 1)) = 0.0
        _out_scattering_ambient ("Outscattering ambients", Range(0, 1)) = 0.0
        
        [Header(Weather Map)]
        [Space(10)]
        _mapScale ("Weather map scale", float) = 1.0
        
        [Header(Misc)]
        [Space(10)]
        _NumStep ("Number of Steps", Range(0, 256)) = 100
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
            #include "Lighting.cginc"

            // Custom Includes
            #include "cginc/Ray.cginc"
            #include "cginc/Volumetric.cginc"
            #include "cginc/Shapes.cginc"
            #include  "cginc/NN_functions.cginc"
            
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

            Texture2D<float4> Weather_tex;
            SamplerState samplerWeather_tex;
            
            float4 _CloudOffset, _DetailOffset;
            float3 _BoundsMin, _BoundsMax;
            float _CloudScale, _DensityThreshold, _DensityMulti, _mapScale, _globalCoverage, _DetailScale;
            float _in_scattering, _out_scattering, _in_out_scattering, _silver_line_intensity, _silver_line_exp;
            float _cloud_beer, _density_to_sun, _cloud_attenuation, _out_scattering_ambient, _param;
            int _NumStep;
            
            uniform float3 fc1_weights_1;
            uniform float3 fc1_weights_2;
            uniform float3 fc1_weights_3;
            uniform float3 fc1_bias;
            
            uniform float3 fc2_weights_1;
            uniform float3 fc2_weights_2;
            uniform float3 fc2_weights_3;
            uniform float3 fc2_bias;

            uniform float3 fc3_weights_1;
            uniform float3 fc3_weights_2;
            uniform float3 fc3_weights_3;
            uniform float3 fc3_bias;

            uniform float3 fc4_weights_1;
            uniform float3 fc4_bias;

            // float sample_density_from_nn(float3 inputs)
            // {
            //     // Convert input into an array for easier access with functions
            //     float position[3] = {inputs.x, inputs.y, inputs.z};
            //
            //     float weight_1[3][3] =
            //     {
            //         {_weights[0], _weights[1], _weights[2]},
            //         {_weights[3], _weights[4], _weights[5]},
            //         {_weights[6], _weights[7], _weights[8]}
            //     };
            //     float bias_1[3] = {_weights[9], _weights[10], _weights[11]};
            //
            //     float weight_2[3][3] =
            //     {
            //         {_weights[12], _weights[13], _weights[14]},
            //         {_weights[15], _weights[16], _weights[17]},
            //         {_weights[18], _weights[19], _weights[20]},
            //     };
            //     float bias_2[3] = {_weights[21], _weights[22], _weights[23]};
            //
            //     float weights_3[3][3] =
            //     {
            //         {_weights[24], _weights[25], _weights[26]},
            //         {_weights[27], _weights[28], _weights[29]},
            //         {_weights[30], _weights[31], _weights[32]},
            //     };
            //     float bias_3[3] = {_weights[33], _weights[34], _weights[35]};
            //
            //     float weights_4[3] = {_weights[36], _weights[37], _weights[38]};
            //     float bias_4 = _weights[39];
            //     
            //     float node_1[3] = {0.0f, 0.0f, 0.0f};
            //     float node_2[3] = {0.0f, 0.0f, 0.0f};
            //     float node_3[3] = {0.0f, 0.0f, 0.0f};
            //
            //     //if (inputs.x > 0.5f) return 0.0f;
            //     
            //     calculate_layer(position, node_1, weight_1, bias_1);
            //     calculate_layer(node_1, node_2, weight_2, bias_2);
            //     calculate_layer(node_2, node_3, weights_3, bias_3);
            //     return calculate_output(position, weights_4, bias_4);
            // }

            // float sample_density(float3 position)
            // {
            //     float4 local_position = mul(unity_WorldToObject, float4(position, 1.0));
            //     const float density = sample_density_from_nn(local_position.xyz);
            //     return density * _DensityMulti;
            // }

            float hard_code_density(float3 position)
            {
                // Convert the world space position into local space
                float4 world_to_local = mul(unity_WorldToObject, float4(position, 1.0f));
                
                // Input -> hidden layer 1
                const float n_1 = relu(fc1_weights_1.x * world_to_local.x + fc1_weights_1.y * world_to_local.y + fc1_weights_1.z * world_to_local.z + fc1_bias.x);
                const float n_2 = relu(fc1_weights_2.x * world_to_local.x + fc1_weights_2.y * world_to_local.y + fc1_weights_2.z * world_to_local.z + fc1_bias.y);
                const float n_3 = relu(fc1_weights_3.x * world_to_local.x + fc1_weights_3.y * world_to_local.y + fc1_weights_3.z * world_to_local.z + fc1_bias.z);

                // Hidden layer 1 -> hidden layer 2
                const float n_4 = relu(fc2_weights_1.x * n_1 + fc2_weights_1.y * n_2 + fc2_weights_1.z * n_3 + fc2_bias.x);
                const float n_5 = relu(fc2_weights_2.x * n_1 + fc2_weights_2.y * n_2 + fc2_weights_2.z * n_3 + fc2_bias.y);
                const float n_6 = relu(fc2_weights_3.x * n_1 + fc2_weights_3.y * n_2 + fc2_weights_3.z * n_3 + fc2_bias.z);

                // Hidden layer 2 -> hidden layer 3
                const float n_7 = relu(fc3_weights_1.x * n_4 + fc3_weights_1.y * n_5 + fc3_weights_1.z * n_6 + fc3_bias.x);
                const float n_8 = relu(fc3_weights_2.x * n_4 + fc3_weights_2.y * n_5 + fc3_weights_2.z * n_6 + fc3_bias.y);
                const float n_9 = relu(fc3_weights_3.x * n_4 + fc3_weights_3.y * n_5 + fc3_weights_3.z * n_6 + fc3_bias.z);

                // Hidden layer 3 -> output
                const float n_10 = relu(fc4_weights_1.x * n_7 + fc4_weights_1.y * n_8 + fc4_weights_1.z * n_9 + fc4_bias);
                
                return n_10 * _DensityMulti;
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
                box.size = _BoundsMax - _BoundsMin;
                box.center = (_BoundsMin + _BoundsMax) * 0.5f;
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
                    // Sample position along the view direction
                    const float3 sample_position = entry_point + primary_ray.direction * dist_travelled;

                    // Sample density
                    //const float density = sample_density(sample_position) * step_size;
                    const float density = hard_code_density(sample_position) * step_size;

                    //if (density < 0.009f) col = float4(1.0f, 0.0f, 0.0f, 1.0f);
                    //if (density > 0.009f) col = float4(0.0f, 1.0f, 0.0f, 1.0f);
                    
                    // if (dist_inside_box > 0.0f)
                    // {
                    //     //float3 sample_position = entry_point + primary_ray.direction * dist_travelled;
                    //     //col = mul(unity_WorldToObject, float4(entry_point + primary_ray.direction * dist_travelled, 1.0f));
                    //     //col = float4(sample_position, 1.0f);
                    //     //const float density = sample_density(sample_position) * step_size;
                    //     //col = density;
                    // }
                    
                    total_density += density;
                    dist_travelled += step_size;
                }

                //return mul(unity_WorldToObject, float4(entry_point + primary_ray.direction * dist_travelled, 1.0f));
                
                //if (col.r < 0.0f) return -col.r;
                //if (col.g < 0.0f) return -col.g;
                //if (col.b < 0.0f) return -col.b;
                //return float4(tex2D(_MainTex, i.uv).rgb + col.r, 1.0f);
                
                // float3 position = entry_point + primary_ray.direction * 1.0f;
                // return mul(unity_WorldToObject, float4(position, 1.0));
                
                // Combining outputs
                const float3 background_color = tex2D(_MainTex, i.uv).rgb;
                return float4(background_color * exp(-total_density) + col.rgb, 1.0f);
            }
            ENDCG
        }
    }
}
