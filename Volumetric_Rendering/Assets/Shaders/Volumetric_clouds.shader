Shader "Hidden/Volumetric_clouds"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            // Helper functions

            // https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
            float2 ray_box_dist(const float3 bound_min, const float3 bound_max, const float3 ray_origin, const float3 ray_dir)
            {
                const float3 t0 = (bound_min - ray_origin) / ray_dir;
                const float3 t1 = (bound_max - ray_origin) / ray_dir;

                const float3 t_min = min(t0, t1);
                const float3 t_max = max(t0, t1);

                const float dist_a = max(max(t_min.x, t_min.y), t_min.z);
                const float dist_b = min(t_max.x, min(t_max.y, t_max.z));

                const float dist_to_box = max(0, dist_a);
                const float dist_inside_box = max(0, dist_b - dist_to_box);

                return float2(dist_to_box, dist_inside_box);
            }

            
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
            float3 _BoundsMin, _BoundsMax;

            fixed4 frag (const v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);
                const float3 ray_origin = _WorldSpaceCameraPos;
                const float3 ray_dir = normalize(i.view_vector);

                const float non_linear_depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
                const float z_depth = LinearEyeDepth(non_linear_depth) * length(i.view_vector);
                
                const float2 ray_box = ray_box_dist(_BoundsMin, _BoundsMax, ray_origin, ray_dir);
                const float dist_to_box = ray_box.x;
                const float dist_inside_box = ray_box.y;

                const bool ray_hit = dist_inside_box > 0 && dist_to_box < z_depth;
                
                if (ray_hit)
                {
                    col = 0;
                }
                return col;
            }
            ENDCG
        }
    }
}
