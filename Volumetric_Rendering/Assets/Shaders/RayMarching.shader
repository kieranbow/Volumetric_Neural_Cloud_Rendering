Shader "RayMarching"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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

			// -----------------------------------------------
			// Structures
			struct Ray
			{
				float3 origin;
				float3 direction;
			};

            struct appdata // Vertex struct
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f // Pixel struct
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
				float3 viewVector : TEXCOORD1;
            };

			// -----------------------------------------------
			// Function
			float2 rayMarchBox(float3 _min, float3 _max, float3 _rayOrigin, float3 _rayDirection)
			{
				float3 t0 = (_min - _rayOrigin) / _rayDirection;
				float3 t1 = (_max - _rayOrigin) / _rayDirection;
				float3 tmin = min(t0, t1);
				float3 tmax = max(t0, t1);

				float distance_A = max(max(tmin.x, tmin.y), tmin.z);
				float distance_B = max(tmax.x, min(tmax.y, tmax.z));

				float dstToBox = max(0, distance_A);
				float distInsideBox = max(0, distance_B - dstToBox);
				return float2(dstToBox, distInsideBox);
			}


			// -----------------------------------------------
			// Vertex Shader
            v2f vert (appdata v)
            {
                v2f output;
				output.vertex = UnityObjectToClipPos(v.vertex);
				output.uv = v.uv;
				float3 view = mul(unity_CameraInvProjection, float4(v.uv * 2 - 1, 0, -1));
				output.viewVector = mul(unity_CameraToWorld, float4(view, 0));
                return output;
            }

			// -----------------------------------------------
			// Samplers
            sampler2D _MainTex;
			sampler2D _CameraDepthTexture;


			// -----------------------------------------------
			// Pixel / Fragment Shader
            fixed4 frag (v2f input) : SV_Target
            {
				float nonLinearDepth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, input.uv);
				float depth = LinearEyeDepth(nonLinearDepth) * length(input.viewVector);

                //fixed4 col = tex2D(_MainTex, input.uv);
                // col.rgb = 1 - col.rgb;

                return Linear01Depth(nonLinearDepth);
            }
            ENDCG
        }
    }
}
