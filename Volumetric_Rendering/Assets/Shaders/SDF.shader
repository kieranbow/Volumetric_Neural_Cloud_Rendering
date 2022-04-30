Shader "Custom/SDF"
{
    Properties
    {
        [HideInInspector] _MainTex ("Texture", 2D) = "white" {}
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

            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            #include "cginc/Ray.cginc"
            #include "cginc/Shapes.cginc"

            float sdSphere(float3 p, float s )
            {
              return length(p)-s;
            }
            
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
                float4 screenPos  : TEXCOORD2;
            };

            v2f vert (const appdata v)
            {
                 // Standard transforming of vertices to clip space
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;

                // The view vector is the direction at which the camera is looking
                float3 view_vector = mul(unity_CameraInvProjection, float4(v.uv * 2.0f - 1.0f, 0.0f, -1.0f));
                o.view_vector = mul(unity_CameraToWorld, float4(view_vector, 0.0f));

                o.screenPos = ComputeScreenPos(o.vertex);
                
                return o;
            }

            sampler2D _MainTex, _CameraDepthTexture;

            float3 _BoundsMin, _BoundsMax;
            
            fixed4 frag (const v2f i) : SV_Target
            {
                // Create a primary ray from the camera
                Ray primary_ray;
                primary_ray.origin = _WorldSpaceCameraPos;
                primary_ray.direction = normalize(i.view_vector);
                
                // Get the depth texture and convert it from eye depth to linear depth
                const float non_linear_depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
                const float z_depth = LinearEyeDepth(non_linear_depth) * length(i.view_vector);
                
                // Assign the box data from the bounding box in the c# script
                box box;
                box.size = _BoundsMax - _BoundsMin;
                box.center = (_BoundsMin + _BoundsMax) * 0.5f;
                box.bound_max = _BoundsMax;
                box.bound_min = _BoundsMin;

                const float2 ray_box = ray_box_dist(box.bound_min, box.bound_max, primary_ray);
                const float dist_to_box = ray_box.x;
                const float dist_inside_box = ray_box.y;

                const float3 entry_point = primary_ray.origin + primary_ray.direction * dist_to_box;

                const float3 background = tex2D(_MainTex, i.uv);
                float4 colour = float4(0.0f, 0.0f, 0.0f, 1.0f);
                if (dist_inside_box > 0.0f)
                {
                    float t = 0.0f;
                    for (int step = 0; step < 100; step++)
                    {
                        const float3 worToLoc = mul(unity_WorldToObject, float4(entry_point + primary_ray.direction * t, 1.0f)).rgb;
                        const float sdf = sdSphere(worToLoc, 1.0f);
                        if (sdf < 0.0001f || t > 5.0f) break;
                        t += sdf;
                    }
                    if (t < 5.0f)
                    {
                        colour.r = 1.0f;
                    }
                }
                return float4(background + colour, 1.0f);
            }
            ENDCG
        }
    }
}
