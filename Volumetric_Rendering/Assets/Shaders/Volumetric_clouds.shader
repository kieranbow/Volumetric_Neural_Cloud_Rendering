Shader "Custom/Volumetric_clouds"
{
    Properties
    {
        _MainTex            ("Texture", 2D)                         = "white" {}
        _CloudScale         ("Cloud Scale", Range(1.0, 100.0))      = 50.0
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

            // Custom Includes
            #include "cginc/Ray.cginc"
            #include "cginc/Volumetric.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float3 view_vector : TEXCOORD1;
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
                const float uvw = position * _CloudScale * 0.001 + _CloudOffset * 0.01;
                float4 shape = volume_texture_3d.SampleLevel(samplervolume_texture_3d, uvw, 0);
                return max(shape.r - _DensityThreshold, 0.0f) * _DensityMulti;
            }
            
            fixed4 frag (const v2f i) : SV_Target
            {
                // Main render texture
                fixed4 col = tex2D(_MainTex, i.uv);

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
                    float3 sample_position = ray.origin + ray.direction * (dist_to_box + dist_travelled);
                    total_density += sample_density(sample_position) * step_size;
                    dist_travelled += step_size;
                }
                
                return col * exp(-total_density);
            }
            ENDCG
        }
    }
}
