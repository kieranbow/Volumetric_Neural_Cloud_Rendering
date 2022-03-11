Shader "Custom/Wireframe"
{
	Properties
	{
		[MainColor] _MainColor ("Surface Color", Color) = (1,1,1,1)
		_Alpha ("Transparency", Range(0, 1)) = 1.0
		_WireFrameColor ("WireFrame Color", Color) = (1,1,1,1)
		[Toggle] _NoSurf("Toggle surface Color", Range(0, 1)) = 1
	}
	SubShader
	{
		Tags 
		{ 
			"RenderType" = "Transparent" 
			"IgnoreProjector" = "True"
			"Queue" = "Transparent"
		}
		ZWrite Off
		Blend SrcAlpha OneMinusSrcAlpha
        Cull back 
		LOD 100

		Pass
		{
			CGPROGRAM
			#pragma target 4.0
			#pragma vertex vert
			#pragma geometry geo
			#pragma fragment frag

			#include "UnityCG.cginc"

			float4 _MainColor, _WireFrameColor;
			float _Alpha, _NoSurf;
			
			struct vertex_output
			{
				float4 position : SV_POSITION; // Vertex local position
				float3 normal : NORMAL; // Vertex normal
				float4 uv : TEXCOORD0; // UV coord of vertex
				float3 world_position : TEXCOORD1; // Vertex world position
			};
			
			vertex_output vert (appdata_base v)
			{
				vertex_output o;
				o.position			= UnityObjectToClipPos(v.vertex);
				o.normal			= UnityObjectToWorldNormal(v.normal);
				o.uv				= float4(v.texcoord.xy, 0.0f, 0.0f);
				o.world_position	= mul(unity_ObjectToWorld, v.vertex).xyz;
				return o;
			}

			struct geo_data
			{
				vertex_output data;
				float2 baryCentricCoords : TEXCOORD2;
			};

			[maxvertexcount(3)]
			void geo(triangle vertex_output i[3], inout TriangleStream<geo_data> stream)
			{
				// Get world position of the three vertices
				const float3 p0 = i[0].world_position.xyz;
				const float3 p1 = i[1].world_position.xyz;
				const float3 p2 = i[2].world_position.xyz;

				// Calculate the triangle normal 
				const float3 triangle_normal = normalize(cross(p1 - p0, p2 - p0));

				geo_data g0, g1, g2;
				g0.data = i[0];
				g1.data = i[1];
				g2.data = i[2];

				// Give each vertex a barycentric coord
				g0.baryCentricCoords = float2(1.0f, 0.0f);
				g1.baryCentricCoords = float2(0.0f, 1.0f);
				g2.baryCentricCoords = float2(0.0f, 0.0f);
				
				// Replace vertex normals with triangle normals
				i[0].normal = triangle_normal;
				i[1].normal = triangle_normal;
				i[2].normal = triangle_normal;
				
				// Append vertex data to stream
				stream.Append(g0);
				stream.Append(g1);
				stream.Append(g2);
			}
			
			fixed4 frag (geo_data i) : SV_Target
			{
				fixed3 barys;
				barys.xy = i.baryCentricCoords;
				barys.z = 1.0f - barys.x - barys.y;

				fixed min_bary = min(barys.x, min(barys.y, barys.z));
				fixed detla = fwidth(min_bary);
				min_bary = smoothstep(0.0f, detla, min_bary);

				fixed4 output_with_surf = fixed4(lerp(_WireFrameColor.rgb, _MainColor.rgb, min_bary), _Alpha);
				fixed4 output_without_surf = fixed4(_WireFrameColor.rgb, 1.0f - min_bary);
				
				return lerp(output_without_surf, output_with_surf, _NoSurf);
			}
			ENDCG
		}
	}
}