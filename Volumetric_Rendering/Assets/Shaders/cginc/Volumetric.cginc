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
    // const float g2 = g * g;
    return ((1.0f - g * g) / pow(1.0f + g * g - 2.0f * (g * g) * cos_theta, 1.5)) / 4.0f * pi;
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

float get_coverage(float4 weather_map)
{
    return weather_map.r;
}

float get_cloud_type(float4 weather_map)
{
    return weather_map.g;
}

float get_precipitation(float4 weather_map)
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
    return max(get_coverage(weather_map), saturate(global_coverage - 0.5f) * get_cloud_type(weather_map) * 2.0f);
}

float alter_shape_height(const float height_percent, const float4 weather_map)
{
    const float bottom_round_clouds = saturate(remap(height_percent, MIN, 0.07f, MIN, MAX));
    const float top_round_clouds = saturate(remap(height_percent, get_precipitation(weather_map) * 0.2f, weather_map.b, MAX, MIN));
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