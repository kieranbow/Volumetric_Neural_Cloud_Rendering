Shader "Hidden/Density"
{
	Properties
	{
        [HideInInspector] _MainTex ("Texture", 2D) = "white" {}
        [HideInInspector] _cloudsTex ("Cloud texture", 2D) = "white" {}
        
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
            #include "cginc/Volumetric.cginc"
            #include "cginc/Shapes.cginc"

            #define MIP_LEVEL 0.0f
			
			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv			: TEXCOORD0;
				float4 vertex		: SV_POSITION;
				float3 view_vector	: TEXCOORD1;
                float4 screenPos	: TEXCOORD2;
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

            // Texture declares
            Texture3D<float4> Shape_tex;
            Texture3D<float4> Noise_tex;
            Texture2D<float4> Weather_tex;
            Texture2D<float4> blueNoise_tex;
            
            // Sample States
            SamplerState samplerShape_tex;
            SamplerState samplerNoise_tex;
            SamplerState samplerWeather_tex;
            SamplerState samplerblueNoise_tex;
			
			fixed4 frag (v2f i) : SV_Target
			{
				fixed4 col = tex2D(_MainTex, i.uv);
				// just invert the colors
				col = 1 - col;
				return col;
			}
			ENDCG
		}
	}
}