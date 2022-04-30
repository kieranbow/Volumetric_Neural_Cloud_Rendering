Shader "Custom/Volumetric_clouds"
{
    Properties
    {
        [HideInInspector] _MainTex ("Texture", 2D) = "white" {}
        [HideInInspector] _cloudsTex ("Cloud texture", 2D) = "white" {}
        
        [Header(Base Shape)]
        [Space(10)]
        _CloudScale ("Cloud Scale", Range(1.0, 1000.0)) = 10.0
        _CloudOffset ("Cloud Offset", Vector) = (0,0,0,0)
        
        [Header(Shape Detail)]
        [Space(10)]
        _DetailScale ("Detail Scale", Range(0, 500)) = 50.0
        _DetailOffset("Detail offset", Vector) = (0,0,0,0)
        
        [Header(Cloud Density)]
        [Space(10)]
        _DensityMulti ("Density Multiplier", Range(0, 100)) = 5.0
        
        [Header(Light Scattering)]
        [Space(10)]
        _in_scattering ("In Scattering", Range(0, 1)) = 0.0
        _out_scattering ("Out Scattering", Range(0, 1)) = 1.0
        _in_scattering_intensity ("In scattering Intensity", Range(0, 1)) = 1.0
        _in_scattering_exp ("In scattering Exponent", Range(0, 20)) = 0.0
        _io_bias ("In-Out Scattering bias", Range(0, 1)) = 0.3
        _out_scattering_ambient ("Out-Scattering ao", Range(0, 1)) = 0.5
        
        [Header(Cloud Appearance)]
        [Space(10)]
        _globalCoverage ("Cloud coverage", Range(0, 1)) = 0.5
        _cloud_attenuation ("Attenuation", Range(0, 1)) = 0.0
        _BeerAmount ("Beer amount", float) = 1.0
        
        [Header(Weather Map)]
        [Space(10)]
        _mapScale ("Weather map scale", float) = 1.0
        
        [Header(Misc)]
        [Space(10)]
        _NumStep ("Number of Steps", Range(0, 256)) = 100
    }
    SubShader
    {
        // No culling or depth
        Cull Off 
        ZWrite Off 
        ZTest Always 

        // This pass will render the clouds
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

            #define MIP_LEVEL 0.0f
            
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

            // In order for sample states in unity to work, they need to be paired together
            // like this: Sampler + Texture name. More is explained here https://docs.unity3d.com/Manual/SL-SamplerStates.html
            
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

            float4 _CloudOffset, _DetailOffset;
            float3 _BoundsMin, _BoundsMax;
            float _CloudScale, _DensityMulti, _mapScale, _globalCoverage, _DetailScale;
            float _in_scattering, _out_scattering, _in_scattering_intensity, _in_scattering_exp, _io_bias;
            float _cloud_attenuation, _out_scattering_ambient, _BeerAmount;
            int _NumStep;
            
            float sample_density(float3 position, const float height_percent)
            {
                // Create uvw using the sample position and the scale and offset of the cloud texture
                const float3 uvw = position * _CloudScale * 0.001f + _CloudOffset * 0.01f;
                
                // Sample the weather map
                const float4 weather_map        = Weather_tex.SampleLevel(samplerWeather_tex, position.xz / _mapScale, MIP_LEVEL);
                const float weather_map_control = normalize_weather_map(weather_map, _globalCoverage);

                // Alter the cloud base shape using a altering shape height function
                const float shape_altering = alter_shape_height(height_percent, weather_map);

                // Alter the cloud's density using a altering density height Function
                const float density_altering = alter_density_height(height_percent, weather_map, _globalCoverage);
                
                // Sample 3D noise
                float4 shape_noise = Shape_tex.SampleLevel(samplerShape_tex, uvw, MIP_LEVEL);
                
                // Create the base cloud shape
                const float SNsample    = remap(shape_noise.r, generate_fbm(shape_noise) - MAX, MAX, MIN, MAX);
                const float SN          = saturate(remap(SNsample * shape_altering, MAX - _globalCoverage * weather_map_control, MAX, MIN, MAX)) * density_altering;

                // Create the detail noise shape
                if (SN > MIN)
                {
                    const float3 detail_uvw         = position * _DetailScale * 0.001f + _DetailOffset * 0.01f;
                    const float4 detail_noise       = Noise_tex.SampleLevel(samplerNoise_tex, detail_uvw, MIP_LEVEL);
                    const float detail_noise_fbm    = generate_fbm(detail_noise);
                    
                    const float e = -(_globalCoverage * 0.75f);
                    const float detail_noise_mod = 0.35f * e * lerp(detail_noise_fbm, MAX - detail_noise_fbm, saturate(height_percent * 5.0f));
                    const float density = saturate(remap(SN, detail_noise_mod, MAX, MIN, MAX)) * density_altering;
                    return density * _DensityMulti;
                }
                return 0;
            }

            float sample_light(const float cos_theta, const float height_percentage, const float density)
            {
                // Lights attenuation inside the cloud
                const float attenuation = light_attenuation(density, cos_theta, _BeerAmount, _cloud_attenuation);
                
                // In out scattering
                const float io_scattering = scattering(cos_theta, _in_scattering, _out_scattering, _in_scattering_intensity, _in_scattering_exp, _io_bias);
                
                // Out-Scattering ambient occlusion
                const float o_scattering_ao = out_scattering_ao(height_percentage, density, _out_scattering_ambient);
                
                return attenuation * io_scattering * o_scattering_ao;
            }

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

                // Sample the blue noise texture to create a random offset
                const float blueNoise = blueNoise_tex.SampleLevel(samplerblueNoise_tex, i.screenPos.xy, MIP_LEVEL);
                
				// Check if a ray has intersected the bounding box and return
				// - The distance to the bounding box
				// - The distance inside the bounding box
                const float2 ray_box = ray_box_dist(box.bound_min, box.bound_max, primary_ray);
                const float dist_to_box = (blueNoise - 0.5f) + ray_box.x;
                const float dist_inside_box = ray_box.y;

                // Create the step size from the distance inside the box / the number of steps that can
				// be taken inside the bounding box
                const float step_size = dist_inside_box / _NumStep;

                // This is mainly use to correct the depth
                const float dist_limit = min(z_depth - dist_to_box, dist_inside_box);

                // If a ray has intersected the bounding box, an entry point is created which will
				// be the start of the ray marching
                const float3 entry_point = primary_ray.origin + primary_ray.direction * dist_to_box;

                float transmittance = 1.0f;
                float3 total_light = float3(0.0f, 0.0f, 0.0f);

                float t = 0.0f;
                while (t < dist_limit)
                {
                    // Sample a new position along the view direction
                    const float3 sample_position = entry_point + primary_ray.direction * t;
                    
                    // Height Percentage
                    const float height_percent = calculate_height_percentage(sample_position, box.bound_min, box.size);

                    // Sample the density at each point
                    const float density = sample_density(sample_position, height_percent) * step_size;

                    // Only add lighting if the density is greater than 0
                    if(density > 0.0f)
                    {
                        // Here cos theta is the angle between the ray direction and light direction
                        const float cos_theta = dot(primary_ray.direction, _WorldSpaceLightPos0.xyz);

                        // Sample the light
                        const float light = sample_light(cos_theta, height_percent, density);
                        total_light += density * step_size * transmittance * light;

                        transmittance *= exp(-density * step_size * _cloud_attenuation);
                        
                        if (transmittance < 0.01f || transmittance > 1.0f) break;
                    }

                    // Once sampling the density and light is finished, update t so that the sample position
					// updates along the viewing direction
                    t += step_size;
                }

                // Combining outputs
                const float3 cloud_color = total_light * _LightColor0.rgb;
                const float3 final_color = float3(0.0f, 0.0f, 0.0f) * transmittance + cloud_color;
                return float4(final_color, 1.0f);
            }
            ENDCG
        }
        
        // This pass combines the rendered clouds with a render from the camera
        Pass 
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            // Includes
            #include "UnityCG.cginc"

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
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;

                float3 view_vector = mul(unity_CameraInvProjection, float4(v.uv * 2.0f - 1.0f, 0.0f, -1.0f));
                o.view_vector = mul(unity_CameraToWorld, float4(view_vector, 0.0f));
                o.screenPos = ComputeScreenPos(o.vertex);

                return o;
            }

            sampler2D _MainTex, _cloudsTex;

            fixed4 frag (const v2f i) : SV_Target
            {
                // Combining outputs
                const float3 background_color = tex2D(_MainTex, i.uv).rgb;
                const float cloud_colour = tex2D(_cloudsTex, i.uv).r;

                return float4(background_color + (cloud_colour), 1.0f);
            }
            
            ENDCG
        }
    }
}
