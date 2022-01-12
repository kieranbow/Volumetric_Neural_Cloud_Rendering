// -----------------------------------------------
// Helper Function
static float pi = 3.14159265359f;

struct light_data
{
    float in_scattering;
    float out_scattering;
    float silver_lining_intensity;
    float silver_lining_exp;
};

// Function from: http://advances.realtimerendering.com/s2017/Nubis%20-%20Authoring%20Realtime%20Volumetric%20Cloudscapes%20with%20the%20Decima%20Engine%20-%20Final%20.pdf
float remap(const float value, const float old_min, const float old_max, const float new_min, const float new_max)
{
    //  return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min);
    return new_min + (((value - old_min) / (old_max - old_min)) * (new_max - new_min));
}

float4 blend_under(float4 color, const float4 new_color)
{
    color.rgb += (1.0 - color.a) * new_color.a * new_color.rgb;
    color.a += (1.0 - color.a) * new_color.a;
    return color;
}

float henyey_Greenstein(const float cos_theta, const float g)
{
    const float g2 = g * g;
    return ((1.0f - g2) / pow(1.0f + g2 - 2.0f * g * cos_theta, 1.5)) / 4.0f * pi;
}

float beer_law(const float density)
{
    return exp(-density);
}

float in_out_scattering(const float cos_angle, const light_data light_data, const float contrast)
{
    const float hg1 = henyey_Greenstein(cos_angle, light_data.in_scattering);
    const float hg2 = light_data.silver_lining_intensity * pow(saturate(cos_angle), light_data.silver_lining_exp);

    const float hg_in_scattering = max(hg1, hg2);
    const float hg_out_scattering = henyey_Greenstein(cos_angle, -light_data.out_scattering);

    return lerp(hg_in_scattering, hg_out_scattering, contrast);
}

float height_signal(float height_percent, float4 weather_map)
{
    float height = saturate(remap(height_percent, 0.0f, 0.07f, 0.0f, 1.0f));
    float end_height = saturate(weather_map.b + 0.12);
    height *= saturate(remap(height_percent, end_height * 0.2, end_height, 1.0f, 0.0f));
    height = pow(height, saturate(remap(height_percent, 0.65f, 0.95f, 1.0f, 0.5f * 0.8f)));
    return height;
}

float4 generate_fbm(float4 noise)
{
    return (noise.g * 0.625f) + (noise.b * 0.25f) + (noise.a * 0.125f);
}

float4 gradient_mix(const float cloud_type)
{
    const float4 stratus_gradient = float4(0.02f, 0.05f, 0.09f, 0.11f);
    const float4 stratocumulus_gradient = float4(0.02f, 0.2f, 0.48f, 0.625f);
    const float4 cumulus_gradient = float4(0.01f, 0.0625f, 0.78f, 1.0f);
    
    const float stratus = 1.0f - saturate(cloud_type * 2.0f);
    const float stratocumulus = 1.0f - abs(cloud_type - 0.5f) * 2.0f;
    const float cumulus = saturate(cloud_type - 0.5f) * 2.0f;

    return (stratus_gradient * stratus) + (stratocumulus_gradient * stratocumulus) + (cumulus_gradient * cumulus);
}

float density_height_gradient(float height_fraction, float cloud_type)
{
    float4 cloud_gradient = gradient_mix(cloud_type);
    return smoothstep(cloud_gradient.x, cloud_gradient.y, height_fraction) - smoothstep(cloud_gradient.z, cloud_gradient.w, height_fraction);
}