Shader "Unlit/Volumetric_test"
{
    Properties
    {
        _MainTex ("Texture", 3D) = "white" {}
		_Alpha ("Alpha", float) = 0.02
		_StepSize("Step Size", float) = 0.01
    }
    SubShader
    {
        Tags 
		{ 
			"RenderType" = "Transparent" 
			"Queue" = "Transparent"
		}
		Blend One OneMinusSrcAlpha
        LOD 100

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
            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
				float3 objectVertex : TEXCOORD0;
				float3 vectorToSurface : TEXCOORD1;
            };
			
			// -----------------------------------------------
			// Helper Function
			float4 BlendUnder(float4 color, float4 newColor)
			{
				color.rgb += (1.0 - color.a) * newColor.a * newColor.rgb;
				color.a += (1.0 - color.a) * newColor.a;
				return color;
			}


			#define EPSILON 0.00001f
			#define MAX_STEP_COUNT 128

            sampler3D _MainTex;
            float4 _MainTex_ST;
			float _Alpha;
			float _StepSize;

			// -----------------------------------------------
			// Vertex_Shader
            v2f vert (appdata v)
            {
                v2f output;
				output.objectVertex = v.vertex;

				float3 worldVertex = mul(unity_ObjectToWorld, v.vertex).xyz;
				output.vectorToSurface = worldVertex - _WorldSpaceCameraPos;

				output.vertex = UnityObjectToClipPos(v.vertex);
                return output;
            }
			// -----------------------------------------------
			// Pixel/fragment_Shader
            fixed4 frag (v2f input) : SV_Target
            {
				float3 offset = float3(0.5f, 0.5f, 0.5f);

				Ray ray;
				ray.origin = input.objectVertex;

				ray.direction = mul(unity_WorldToObject, float4(normalize(input.vectorToSurface), 1));

				float4 color = float4(0.0f, 0.0f, 0.0f, 0.0f);
				float3 samplePosition = ray.origin;

				for (int i = 0; i < MAX_STEP_COUNT; i++)
				{
					if (max(abs(samplePosition.x), max(abs(samplePosition.y), abs(samplePosition.z))) < 0.5f + EPSILON)
					{
						float4 sampleColor = tex3D(_MainTex, samplePosition + offset);
						sampleColor.a *= _Alpha;
						color = BlendUnder(color, sampleColor);
						samplePosition += ray.direction * _StepSize;
					}
				}
				return color;
            }
            ENDCG
        }
    }
}
