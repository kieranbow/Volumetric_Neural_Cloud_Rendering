Shader "Custom/Volumetric_clouds"
{
    Properties
    {
        _MainTex            ("Texture", 2D)                         = "white" {}
        _CloudScale         ("Cloud Scale", Range(1.0, 1000.0))      = 50.0
        _CloudOffset        ("Cloud Offset", Range(0.1, 10.0))      = 1.0
        _DensityThreshold   ("Density Threshold", Range(0, 1))      = 0.5
        _DensityMulti       ("Density Multiplier", float)           = 5.0
        _NumStep            ("Number of Steps", Range(0, 256))      = 100
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always // No culling or depth

        Pass
        {
            CGPROGRAM
            #pragma exclude_renderers d3d11_9x
            
            #pragma vertex vert
            #pragma fragment frag

            // Includes
            #include "UnityCG.cginc"
            #include "Lighting.cginc"

            // Custom Includes
            #include "cginc/Ray.cginc"
            #include "cginc/Volumetric.cginc"

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

            Texture3D<float4> volume_texture_3d;
            SamplerState samplervolume_texture_3d;

            float3 _BoundsMin, _BoundsMax;
            float _CloudScale, _CloudOffset, _DensityThreshold, _DensityMulti;
            int _NumStep;
            
            float sample_density(float3 position)
            {
                const float3 uvw = position * _CloudScale * 0.001f + _CloudOffset * 0.01f;
                float4 shape = volume_texture_3d.SampleLevel(samplervolume_texture_3d, uvw, 0.0f);
                return max(shape.g - _DensityThreshold, 0.0f) * _DensityMulti;
            }

            fixed4 frag (const v2f i) : SV_Target
            {
                float4 col = tex2D(_MainTex, i.uv);
                
                // Depth texture
                const float non_linear_depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
                const float z_depth = LinearEyeDepth(non_linear_depth) * length(i.view_vector);
                
                // Ray
                ray ray;
                ray.origin = _WorldSpaceCameraPos;
                ray.direction = normalize(i.view_vector);
                
                // Ray box intersection
                const float2 ray_box = ray_box_dist(_BoundsMin, _BoundsMax, ray);
                const float dist_to_box = ray_box.x;
                const float dist_inside_box = ray_box.y;
                
                const bool ray_hit = dist_inside_box > 0 && dist_to_box < z_depth;
                
                // if (ray_hit)
                // {
                //     col = 0;
                // }

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

                float transmittance = exp(-total_density);
                return col * transmittance;
                
                // Combining outputs
                //const float3 background_color = tex2D(_MainTex, i.uv).rgb;
                //const float3 cloud_color = total_density * _LightColor0.rgb;
                //const float3 final_color = background_color * beer_law(-total_density, 1.0f) + cloud_color;
                //return float4(final_color, 1.0f);
            }
            ENDCG
        }
    }
}
