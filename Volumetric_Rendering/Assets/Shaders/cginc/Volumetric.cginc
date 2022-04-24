// -----------------------------------------------
// Helper Function
static float pi = 3.14159265359f;
#define MIN 0.0f
#define MAX 1.0f
#define GMIN 0.2f
#define GMAX 0.7f

// Function from: http://advances.realtimerendering.com/s2017/Nubis%20-%20Authoring%20Realtime%20Volumetric%20Cloudscapes%20with%20the%20Decima%20Engine%20-%20Final%20.pdf
float remap(const float value, const float old_min, const float old_max, const float new_min, const float new_max)
{
    return new_min + (((value - old_min) / (old_max - old_min)) * (new_max - new_min));
}

float4 blend_under(float4 color, const float4 new_color)
{
    color.rgb += (MAX - color.a) * new_color.a * new_color.rgb;
    color.a += (MAX - color.a) * new_color.a;
    return color;
}

float henyey_Greenstein(const float cos_theta, const float g)
{
    const float g2 = g * g;
    return ((1.0f - g2) / pow(1.0f + g2 - 2.0f * g2 * cos_theta, 1.5)) / 4.0f * pi;
}

float beer_law(const float density)
{
    return exp(-density);
}

float height_signal(float height_percent, float4 weather_map)
{
    float height = saturate(remap(height_percent, 0.0f, 0.07f, 0.0f, 1.0f));
    float end_height = saturate(weather_map.b + 0.12f);
    height *= saturate(remap(height_percent, end_height * 0.2f, end_height, 1.0f, 0.0f));
    height = pow(height, saturate(remap(height_percent, 0.65f, 0.95f, 1.0f, 0.5f * 0.8f)));
    return height;
}

float generate_fbm(float4 noise)
{
    return noise.g * 0.625f + noise.b * 0.25f + noise.a * 0.125f;
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

float calculate_height_percentage(float3 position, float3 bound_min, float3 bound_size)
{
    return (position.y - bound_min.y) / bound_size.y;
}

float calculate_height_gradient(float height_percentage)
{
    return saturate(
        remap(height_percentage, MIN, GMIN, MIN, MAX)) *
            saturate(remap(height_percentage, MAX, GMAX, MIN, MAX));
}

float get_low_coverage(float4 weather_map)
{
    return weather_map.r;
}

float get_high_coverage(float4 weather_map)
{
    return weather_map.g;
}

float get_cloud_peaks(float4 weather_map)
{
    return weather_map.b;
}

float get_density(const float4 weather_map)
{
    return weather_map.a;
}

float height_fraction(float3 position, float2 cloud_minmax)
{
    return saturate((position.z - cloud_minmax.x) / (cloud_minmax.y + cloud_minmax.x));
}

float density_gradient_height(float height, float cloud_type)
{
    float cumulus = max(remap(height, 0.01f, 0.3f, 0.0f, 1.0f) * remap(height, 0.6f, 0.95f, 1.0f, 0.0f), 0.0f);
    float stratocumulus = max(remap(height, 0.0f, 0.25f, 0.0f, 1.0f) * remap(height, 0.3f, 0.65f, 1.0f, 0.0f), 0.0f);
    float stratus = max(remap(height, 0.0f, 0.1f, 0.0f, 1.0f) * remap(height, 0.2f, 0.3f, 1.0f, 0.0f), 0.0f);

    float a = lerp(stratus, stratocumulus, clamp(cloud_type * 2.0f, 0.0f, 1.0f));
    float b = lerp(stratocumulus, stratus, clamp((cloud_type - 0.5f) * 2.0f, 0.0f, 1.0f));
    return lerp(a, b, cloud_type);
}


float normalize_weather_map(const float4 weather_map, const float global_coverage)
{
    return max(get_low_coverage(weather_map), saturate(global_coverage - 0.5f) * get_high_coverage(weather_map) * 2.0f);
}

float alter_shape_height(const float height_percent, const float4 weather_map)
{
    const float bottom_round_clouds = saturate(remap(height_percent, MIN, 0.07f, MIN, MAX));
    const float top_round_clouds = saturate(remap(height_percent, get_cloud_peaks(weather_map) * 0.2f, get_cloud_peaks(weather_map), MAX, MIN));
    return bottom_round_clouds * top_round_clouds;
}

float alter_density_height(const float height_percent, const float4 weather_map, const float global_density)
{
    const float reduce_density = height_percent * saturate(remap(height_percent, MIN, 0.15f, MIN, MAX));
    const float soft_transition = saturate(remap(height_percent, 0.9f, MAX, MAX, MIN));
    return global_density * reduce_density * soft_transition * get_density(weather_map) * 2.0f;
}

float generate_base_shape(float4 shape)
{
    float base_shape = generate_fbm(shape);
    base_shape = -(1.0f - base_shape);
    return remap(shape.r, base_shape, MAX, MIN, MAX);
}

float light_attenuation(const float density_to_sun, const float cos_theta, const float beer_amount, const float atten_clamp)
{
    const float primary_atten = exp(-beer_amount * density_to_sun);
    const float second_atten = exp(-beer_amount * atten_clamp);

    // Reduces the attenuation clamp when facing the sun
    float reduce_clamp = remap(cos_theta, MIN, MAX, second_atten, second_atten * 0.5f);

    return max(reduce_clamp, primary_atten);
}

float phase(float cos_theta)
{
    float hg = henyey_Greenstein(cos_theta, 0.83f) * 0.5f + henyey_Greenstein(cos_theta, -0.3f) * 0.5f;
    return 0.8f + hg * 0.15f;
}

float scattering(const float cos_theta, const float in_scattering, const float out_scattering, const float ins_intensity, const float ins_exp, const float bias)
{
    // Creates the in-scattering effect using henyey_greenstein
    const float hg1 = henyey_Greenstein(cos_theta, in_scattering);

    // Extra intensity to in-scattering
    const float hg2 = ins_intensity * pow(saturate(cos_theta), ins_exp);

    // Selects between the two in-scattering effects
    const float in_scattering_hg = max(hg1, hg2);

    // Creates out-scattering effect using henyey_Greenstein
    const float out_scattering_hg = henyey_Greenstein(cos_theta, -out_scattering);

    // Return the amount of scattering based of a bias
    return lerp(in_scattering_hg, out_scattering_hg, bias);
}

// Adds a non-physically based ambient occlusion to the clouds to create contours and to prevent
// white blowouts
float out_scattering_ao(const float height_percentage, const float density, const float osa)
{
    const float a = saturate(osa * pow(density, remap(height_percentage, 0.3f, 0.9f, 0.5f, 1.0f)));
    const float b = saturate(pow(remap(height_percentage, MIN, 0.3f, 0.8f, 1.0f), 0.8f));
    return MAX - a * b;
}