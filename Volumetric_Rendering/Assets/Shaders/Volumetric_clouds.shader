Shader "Custom/Volumetric_clouds"
{
    Properties
    {
        [HideInInspector] _MainTex            ("Texture", 2D)       = "white" {}
        _CloudScale         ("Cloud Scale", Range(1.0, 100.0))      = 10.0
        _CloudOffset        ("Cloud Offset", Range(0.1, 10.0))      = 1.0
        _DensityThreshold   ("Density Threshold", Range(0, 1))      = 0.5
        _DensityMulti       ("Density Multiplier", float)           = 5.0
        _NumStep            ("Number of Steps", Range(0, 256))      = 100
        _test               ("test", float) = 1.0
        _cloudtype               ("_cloudtype", float) = 1.0
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
            
            // Sample States
            SamplerState samplerShape_tex;
            SamplerState samplerNoise_tex;

            float3 _BoundsMin, _BoundsMax;
            float _CloudScale, _CloudOffset, _DensityThreshold, _DensityMulti, _test, _cloudtype;
            int _NumStep;
            
            float sample_density(float3 position)
            {
                // current sample position
                const float3 uvw = position * _CloudScale * 0.001f + _CloudOffset * 0.01f;

                const float3 height_fraction = position;
                
                // Sample 3D noise
                float4 shape = Shape_tex.SampleLevel(samplerShape_tex, uvw, 0.0f);
                float4 noise = Noise_tex.SampleLevel(samplerNoise_tex, uvw, 0.0f);

                // Base cloud shape
                float fbm = generate_fbm(shape).r;
                float base_shape = remap(shape.r, 1.0f - fbm, 1.0f, 0.0f, 1.0f);

                return max(base_shape - _DensityThreshold, 0.0f) * _DensityMulti;
            }

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
                    total_density += sample_density(sample_position) * step_size;
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
