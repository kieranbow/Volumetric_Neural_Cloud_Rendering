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

            float weights[252];
            float bias[28];

            float sampleDensity(float3 position)
            {
                // Convert the world space position into local space
                float3 world_to_local = mul(unity_WorldToObject, float4(position, 1.0f)).xyz;
                float input[9] =
                {
                    world_to_local.x,
                    world_to_local.y,
                    world_to_local.z,
                    pow(world_to_local.x, 2),
                    pow(world_to_local.y, 2),
                    pow(world_to_local.z, 2),
                    sin(world_to_local.x * 16 * UNITY_PI),
                    sin(world_to_local.y * 16 * UNITY_PI),
                    sin(world_to_local.z * 16 * UNITY_PI)
                };
                return calculateDensityFromANN(input, weights, bias);
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
                    const float density = sampleDensity(sample_position) * step_size;

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
