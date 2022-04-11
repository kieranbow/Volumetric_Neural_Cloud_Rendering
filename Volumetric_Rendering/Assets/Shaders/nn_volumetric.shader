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

            float weights[8][4] = {
                {-0.20719771087169647f, -0.20719771087169647f, 0.3353896141052246f, -0.20719771087169647f},
                {0.3353896141052246f, 0.4701395332813263f, -0.0421954020857811f, -0.0421954020857811f},
                {0.2507553696632385f, -0.0421954020857811f, 0.2507553696632385f, -0.0421954020857811f},
                {0.2507553696632385f, -0.053198859095573425f, 0.0466652549803257f, 0.0466652549803257f},
                {-0.2594115138053894f, 0.0466652549803257f, -0.2594115138053894f, 0.04129326343536377f},
                {-0.3532930612564087f, -0.3532930612564087f, 0.5556743144989014f, -0.3532930612564087f},
                {0.5556743144989014f, 0.5319416522979736f, 0.010805665515363216f, 0.010805665515363216f},
                {0.14398765563964844f, 0.010805665515363216f, 0.14398765563964844f,  0.13225780427455902f}
            };

            float relu(float x)
            {
                return max(0.0f, x);
            }

            float sigmoid(float x)
            {
                return 1.0f / (1.0f + exp(-x));
            }
            
            float sample_density_from_nn(float3 position, float weights[8][4])
            {
                float neuron = 0.0f;
                for (int column = 0; column < 3; column++)
                {
                    for (int row = 0; row < 3; row++)
                    {
                        neuron += dot(position[row], weights[row][column]);
                    }
                }
                return sigmoid(neuron);
            }

            float sample_density(float3 position, const float height_percent, float weights[8][4])
            {
                // Create uvw using the sample position and the scale and offset of the cloud texture
                const float3 uvw = position * _CloudScale * 0.001f + _CloudOffset * 0.01f;
                
                return sample_density_from_nn(uvw, weights);
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
                    
                    // Sample density
                    total_density += sample_density(sample_position, height_percent, weights) * step_size * _DensityMulti;
                    
                    dist_travelled += step_size;
                }

                // Combining outputs
                const float3 background_color = tex2D(_MainTex, i.uv).rgb;
                const float3 final_color = background_color * exp(-total_density);
                return float4(final_color, 1.0f);
            }
            ENDCG
        }
    }
}
