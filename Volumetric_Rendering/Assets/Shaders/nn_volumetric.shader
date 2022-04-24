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
        _DensityMulti ("Density Multiplier", Range(0, 10)) = 5.0
        
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
        _param ("Cloud brightness", Range(0, 1)) = 0.5
        
        [Header(Weather Map)]
        [Space(10)]
        _mapScale ("Weather map scale", float) = 1.0
        
        [Header(Misc)]
        [Space(10)]
        _NumStep ("Number of Steps", Range(0, 256)) = 100
        
        _X ("X", float) = 0
        _Y ("Y", float) = 0
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
            
            float3 fc1_weights_1;
            float3 fc1_weights_2;
            float3 fc1_weights_3;
            float3 fc1_bias;
            float3 fc2_weights;
            float fc2_bias;

            float _X, _Y;

            Texture2D<float4> _weightTex;
            SamplerState sampler_weightTex;
            //StructuredBuffer<float> weights;

            float sampleWeights(float x, float y)
            {
                float i = clamp(x / 4.0f, 0, 1);
                float j = clamp(y / 4.0f, 0, 1);
                
                return _weightTex.SampleLevel(sampler_weightTex, float2(i, j), 0).r;
            }
            
            float sample_density_from_nn(float3 inputs, Texture2D<float4> weights)
            {
                // Convert input into an array for easier access with functions
                float position[3] = {inputs.x, inputs.y, inputs.z};

                float weight_1[3][3] =
                {
                    {sampleWeights(0.0f, 0.0f), sampleWeights(1.0f, 0.0f), sampleWeights(2.0f, 0.0f)},
                    {sampleWeights(3.0f, 0.0f), sampleWeights(4.0f, 0.0f), sampleWeights(0.0f, 1.0f)},
                    {sampleWeights(1.0f, 2.0f), sampleWeights(2.0f, 3.0f), sampleWeights(2.0f, 4.0f)}
                };

                float bias_1[3] = {0.294, 0.108, 0.316};

                float weight_2[3] = { -0.519, -0.252, 0.108 };
                float bias_2 = -0.269;
                
                float nodes[3] = {0.0f, 0.0f, 0.0f};
                
                calculate_layer(position, nodes, weight_1, bias_1);

                if (inputs.x > 0.5f) return 0.0f;
                
                return calculate_output(nodes, weight_2, bias_2);
            }

            float sample_density(float3 position, Texture2D<float4> weights)
            {
                float4 local_position = mul(UNITY_MATRIX_IT_MV, float4(position, 1.0f));
                const float density = sample_density_from_nn(local_position.xyz, weights);
                return density * _DensityMulti;
            }

            fixed4 frag (v2f i) : SV_Target
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
                while (dist_travelled < dist_limit)
                {
                    // Sample position along the view direction
                    const float3 sample_position = entry_point + primary_ray.direction * dist_travelled;
                    
                    // Height Percentage
                    const float height_percent = calculate_height_percentage(sample_position, box.bound_min, box.size);
                    
                    // float4 weights = _weightTex.SampleLevel(sampler_weightTex, i.uv, 0);
                    
                    // Sample density
                    const float density = sample_density(sample_position, _weightTex) * step_size;

                    total_density += density;
                    dist_travelled += step_size;
                }

                // Combining outputs
                const float3 background_color = tex2D(_MainTex, i.uv).rgb;
                return float4(background_color * exp(-total_density), 1.0f);
            }
            ENDCG
        }
    }
}
