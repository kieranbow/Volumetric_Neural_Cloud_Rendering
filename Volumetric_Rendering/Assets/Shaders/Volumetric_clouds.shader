Shader "Custom/Volumetric_clouds"
{
    Properties
    {
        [HideInInspector] _MainTex ("Texture", 2D) = "white" {}
        
        [Header(Base Shape)]
        [Space(10)]
        _CloudScale ("Cloud Scale", Range(1.0, 100.0)) = 10.0
        _CloudOffset ("Cloud Offset", Vector) = (0,0,0,0)
        
        [Header(Shape Detail)]
        [Space(10)]
        _DetailScale ("Detail Scale", Range(0, 500)) = 50.0
        _DetailOffset("Detail offset", Vector) = (0,0,0,0)
        
        [Header(Cloud Density)]
        [Space(10)]
        _DensityThreshold ("Density Threshold", Range(0, 1)) = 0.5
        _DensityMulti ("Density Multiplier", Range(0, 10)) = 5.0
        
        [Header(Cloud appearance)]
        [Space(10)]
        _globalCoverage ("Cloud coverage", Range(0, 1)) = 0.5
        _in_scattering ("In Scattering", Range(0, 1)) = 0.0
        _out_scattering ("Out Scattering", Range(0, 1)) = 0.0
        _in_out_scattering ("In Out Scattering bias", Range(0, 1)) = 0.0
        _silver_line_intensity ("Silver Lining Intensity", Range(0, 1)) = 0.0
        _silver_line_exp ("Silver Lining Exponent", Range(0, 1)) = 0.0
        _cloud_beer ("Beer amount", Range(0, 1)) = 0.0
        _density_to_sun ("Density to sun", Range(0, 10)) = 0.0
        _cloud_attenuation ("Attenuation", Range(0, 1)) = 0.0
        _out_scattering_ambient ("Outscattering ambients", Range(0, 1)) = 0.0
        _param ("Cloud brightness", Range(0, 1)) = 0.5
        
        [Header(Weather Map)]
        [Space(10)]
        _mapScale ("Weather map scale", float) = 1.0
        
        [Header(Misc)]
        [Space(10)]
        _NumStep ("Number of Steps", Range(0, 256)) = 100
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always // No culling or depth

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
            };

            // -----------------------------------------------------
            // Vertex Shader
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

            // In order for sample states in unity to work, they need to be paired together
            // like this: Sampler + Texture name. More is explained here https://docs.unity3d.com/Manual/SL-SamplerStates.html
            
            // Texture declares
            Texture3D<float4> Shape_tex;
            Texture3D<float4> Noise_tex;
            Texture2D<float4> Weather_tex;
            
            // Sample States
            SamplerState samplerShape_tex;
            SamplerState samplerNoise_tex;
            SamplerState samplerWeather_tex;

            float4 _CloudOffset, _DetailOffset;
            float3 _BoundsMin, _BoundsMax;
            float _CloudScale, _DensityThreshold, _DensityMulti, _mapScale, _globalCoverage, _DetailScale;
            float _in_scattering, _out_scattering, _in_out_scattering, _silver_line_intensity, _silver_line_exp;
            float _cloud_beer, _density_to_sun, _cloud_attenuation, _out_scattering_ambient, _param;
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
                const float SNsample = remap(shape_noise.r, generate_fbm(shape_noise) - MAX, MAX, MIN, MAX);
                const float SN  = saturate(remap(SNsample * shape_altering, MAX - _globalCoverage * weather_map_control, MAX, MIN, MAX)) * shape_altering; // density_altering

                // Create the detail noise shape
                if (SN > MIN)
                {
                    const float3 detail_uvw     = position * _DetailScale * 0.001f + _DetailOffset * 0.01f;
                    const float4 detail_noise   = Noise_tex.SampleLevel(samplerNoise_tex, detail_uvw, MIP_LEVEL);
                    
                    const float detail_noise_fbm = generate_fbm(detail_noise);
                    const float e = -(_globalCoverage * 0.75f);
                    const float detail_noise_mod = 0.35f * e * lerp(detail_noise_fbm, MAX - detail_noise_fbm, saturate(height_percent * 5.0f));
                    const float density = saturate(remap(SN, detail_noise_mod, MAX, MIN, MAX)) * density_altering;
                    return density * _DensityMulti;
                }
                return 0;
            }

            float3 sample_light(float cos_angle, float height_percentage, float density)
            {
                // Attenuation
                float primary = exp(-_cloud_beer * _density_to_sun);
                float secondary = exp(-_cloud_beer * _cloud_attenuation) * 0.7f;
                float check = remap(cos_angle, MIN, MAX, secondary, secondary * 0.5f);
                float attenuation_prob = max(check, primary);
                
                // Ambient Out scattering
                float depth = _out_scattering_ambient * pow(density, remap(height_percentage, 0.3f, 0.9f, 0.5f, MAX));
                float vertical = pow(saturate(remap(height_percentage, MIN, 0.3f, 0.8f, 1.0f)), 0.8f);
                float out_scatter = depth * vertical;
                out_scatter = 1.0 - saturate(out_scatter);
                float ambient_out_scattering = out_scatter;

                // In out scattering
                float hg1 = henyey_Greenstein(cos_angle, _in_scattering);
                float hg2 = _silver_line_intensity * pow(saturate(cos_angle), _silver_line_exp);
                float hg_inscattering = max(hg1, hg2);
                float hg_outscattering = henyey_Greenstein(cos_angle, -_out_scattering);
                float inoutscattering = lerp(hg_inscattering, hg_outscattering, _in_out_scattering);
                float sun_highlight = inoutscattering;

                float attenuation = attenuation_prob * sun_highlight * ambient_out_scattering;
                
                return attenuation;
            }

            // -----------------------------------------------------
            // Pixel Shader
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
                box.size = _BoundsMax - _BoundsMin;
                box.center = (_BoundsMin + _BoundsMax) * 0.5f;
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

                // Cloud lighting
                float cos_angle = dot(primary_ray.direction, _WorldSpaceLightPos0.xyz);

                float transmittance = 1.0f;
                float3 total_light = float3(0.0f, 0.0f, 0.0f);

                float phase = 0.8f + (henyey_Greenstein(cos_angle, 0.85f) * 0.5f + henyey_Greenstein(cos_angle, -0.3f) * 0.5f) * 0.15f;
                
                while (dist_travelled < dist_limit)
                {
                    // Sample position along the view direction
                    const float3 sample_position = entry_point + primary_ray.direction * dist_travelled;
                    
                    // Height Percentage
                    const float height_percent = calculate_height_percentage(sample_position, box.bound_min, box.size);
                    
                    // Sample density
                    float density = sample_density(sample_position, height_percent) * step_size;

                    if(density > 0.0f)
                    {
                        float3 light_march = sample_light(cos_angle, height_percent, density);
                        total_light += density * transmittance * light_march;
                        transmittance *= exp(-density * _cloud_attenuation);
                    
                        if (transmittance < 0.01f) break;
                        if (transmittance > 1.0f)  break;
                    }
                    dist_travelled += step_size;
                }

                // Combining outputs
                const float3 background_color = tex2D(_MainTex, i.uv).rgb;
                const float3 cloud_color = total_light * _LightColor0.rgb;
                const float3 final_color = background_color * transmittance + cloud_color;
                return float4(final_color, 1.0f);
            }
            ENDCG
        }
    }
}
