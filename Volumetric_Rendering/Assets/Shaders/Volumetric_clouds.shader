Shader "Custom/Volumetric_clouds"
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
        _DetailScale ("Detail Scale", Range(0, 100)) = 50.0
        _DetailOffset("Detail offset", Vector) = (0,0,0,0)
        
        [Header(Cloud Density)]
        [Space(10)]
        _DensityThreshold ("Density Threshold", Range(0, 1)) = 0.5
        _DensityMulti ("Density Multiplier", float) = 5.0
        
        [Header(Cloud appearance)]
        [Space(10)]
        _cloudtype ("cloudtype", Range(0, 1)) = 1.0
        _globalCoverage ("Cloud coverage", Range(0, 1)) = 0.5
        
        [Header(Misc)]
        [Space(10)]
        _NumStep ("Number of Steps", Range(0, 256)) = 100
        _mapScale ("Weather map scale", float) = 1.0

    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always // No culling or depth

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

            // -----------------------------------------------------
            // Vertex Shader
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

            // In order for sample states in unity to work, they need to be paired together
            // like this: Sampler + Texture name. More is explained here https://docs.unity3d.com/Manual/SL-SamplerStates.html
            
            // Texture declares
            Texture3D<float4> Shape_tex;
            Texture3D<float4> Noise_tex;
            Texture2D<float4> Weather_tex;
            
            // Sample States
            SamplerState samplerShape_tex;
            SamplerState samplerNoise_tex;
            SamplerState samplerWeather_tex;

            float4 _CloudOffset, _DetailOffset;
            float3 _BoundsMin, _BoundsMax;
            float _CloudScale, _DensityThreshold, _DensityMulti, _mapScale, _cloudtype, _globalCoverage, _DetailScale;
            int _NumStep;
            
            float sample_density(float3 position, box bounding_box)
            {
                // Current sample position
                const float3 uvw = position * _CloudScale * 0.001f + _CloudOffset * 0.01f;
                const float3 detail_uvw = position * _DetailScale * 0.001f + _DetailOffset * 0.01f;

                // Weather map
                float4 sample_weather = Weather_tex.SampleLevel(samplerWeather_tex, position.xz / _mapScale, 0.0f);
                float low_coverage = sample_weather.r;
                float high_coverage = sample_weather.g;
                float peaks = sample_weather.b;
                float density = sample_weather.a;
                float WMc = max(sample_weather.r, saturate(_globalCoverage - 0.5f) * sample_weather.g * 2.0f);

                // Height Percentage
                float height_percent = calculate_height_percentage(position, bounding_box.bound_min, bounding_box.size);

                // Shape Altering height Function
                float SRb = saturate(remap(height_percent, MIN, 0.07f, MIN, MAX));
                float SRt = saturate(remap(height_percent, peaks * 0.2f, peaks, MAX, MIN));
                float SA = SRb * SRt;

                // Density Altering Height Function
                float DRb = height_percent * saturate(remap(height_percent, MIN, 0.15f, MIN, MAX));
                float DRt = saturate(remap(height_percent, 0.9f, MAX, MAX, MIN));
                float DA = _globalCoverage * DRb * DRt * density * 2.0f;
                
                // Sample 3D noise
                float4 shape_noise = Shape_tex.SampleLevel(samplerShape_tex, uvw, 0.0f);
                float4 detail_noise = Noise_tex.SampleLevel(samplerNoise_tex, detail_uvw, 0.0f);

                // Base Shape
                float SNsample = remap(shape_noise.r, generate_fbm(shape_noise) - 1.0f, MAX, MIN, MAX);
                float SN = saturate(remap(SNsample * SA, 1.0f - _globalCoverage * WMc, MAX, MIN, MAX)); //* DA;

                // Detail Noise
                float DNfbm = generate_fbm(detail_noise);
                float e = _globalCoverage * 0.75f;
                float DNmod = 0.35f * e * lerp(DNfbm, 1.0f - DNfbm, saturate(height_percent * 5.0f));
                float d = saturate(remap(SN, DNfbm, MAX, MIN, MAX)) * DA;

                return d;
                
                // const float height_gradient = saturate(remap(height_percent, 0.0f, 0.2f, 0.0f, 1.0f)) * saturate(remap(height_percent, 1.0f, 0.7f, 0.0f, 1.0f));
                // float base_shape = generate_base_shape(shape_noise);
                // base_shape *= alter_shape_height(height_percent, peaks);

                //return max(d - _DensityThreshold, 0.0f) * _DensityMulti;
            }

            // -----------------------------------------------------
            // Pixel Shader
            fixed4 frag (const v2f i) : SV_Target
            {
                // Depth texture
                const float non_linear_depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
                const float z_depth = LinearEyeDepth(non_linear_depth) * length(i.view_vector);
                
                // Ray
                ray ray;
                ray.origin = _WorldSpaceCameraPos;
                ray.direction = normalize(i.view_vector);

                // Box data
                box box;
                box.size = _BoundsMax - _BoundsMin;
                box.center = (_BoundsMin + _BoundsMax) * 0.5f;
                box.bound_max = _BoundsMax;
                box.bound_min = _BoundsMin;
                
                // Ray box intersection
                const float2 ray_box = ray_box_dist(box.bound_min, box.bound_max, ray);
                const float dist_to_box = ray_box.x;
                const float dist_inside_box = ray_box.y;

                float dist_travelled = 0.0f;
                const float step_size = dist_inside_box / _NumStep;
                const float dist_limit = min(z_depth - dist_to_box, dist_inside_box);

                float total_density = 0.0f;
                while (dist_travelled < dist_limit)
                {
                    const float3 sample_position = ray.origin + ray.direction * (dist_to_box + dist_travelled);
                    total_density += sample_density(sample_position, box) * step_size;
                    dist_travelled += step_size;
                }

                // Combining outputs
                const float3 background_color = tex2D(_MainTex, i.uv).rgb;
                const float3 cloud_color = total_density * _LightColor0.rgb;
                const float3 final_color = background_color * beer_law(total_density) + cloud_color;
                return float4(final_color, 1.0f);
            }
            ENDCG
        }
    }
}
