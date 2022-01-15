// -----------------------------------------------
// Helper Function
static float pi = 3.14159265359f;
#define MIN 0.0f
#define MAX 1.0f

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

float getCoverage(float3 weather_map)
{
    return weather_map.r;
}

float getCloud_type(float3 weather_map)
{
    return weather_map.g;
}

float getPrecipitation(float3 weather_map)
{
    return weather_map.b;
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



float normalize_weather_map(float4 weather_map, float global_coverage)
{
    return max(weather_map.r, saturate(global_coverage - 0.5f) * weather_map.g * 2.0f);
}

float alter_shape_height(float height_percent, float weather_height)
{
    float bottom_round_clouds = saturate(remap(height_percent, 0.0f, 0.07f, MIN, MAX));
    float top_round_clouds = saturate(remap(height_percent, weather_height * 0.2f, weather_height, MAX, MIN));
    return bottom_round_clouds * top_round_clouds;
}

float alter_density_height(float height_percent, float weather_density, float global_density)
{
    float reduce_density = height_percent * saturate(remap(height_percent, MIN, 0.15f, MIN, MAX));
    float soft_transition = saturate(remap(height_percent, 0.9f, MAX, MAX, MIN));
    return global_density * reduce_density * soft_transition * weather_density * 2.0f;
}

float generate_base_shape(float4 shape)
{
    float base_shape = generate_fbm(shape);
    base_shape = -(1.0f - base_shape);
    return remap(shape.r, base_shape, MAX, MIN, MAX);
}

float function_name(float shape, float global_coverage, float weather_map, float shape_altered, float density_altered)
{
    float sa = shape_altered;
    float da = density_altered;
    return saturate(remap(shape * sa, 1.0f - global_coverage * weather_map, MAX, MIN, MAX)) * da;
}