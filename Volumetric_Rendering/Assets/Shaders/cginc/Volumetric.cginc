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

float remap(const float value, const float old_min, const float old_max, const float new_min, const float new_max)
{
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min);
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

float beer_law(const float density, const float absorption)
{
    return exp(-density * absorption);
}

float in_out_scattering(const float cos_angle, const light_data light_data, const float contrast)
{
    const float hg1 = henyey_Greenstein(cos_angle, light_data.in_scattering);
    const float hg2 = light_data.silver_lining_intensity * pow(saturate(cos_angle), light_data.silver_lining_exp);

    const float hg_in_scattering = max(hg1, hg2);
    const float hg_out_scattering = henyey_Greenstein(cos_angle, -light_data.out_scattering);

    return lerp(hg_in_scattering, hg_out_scattering, contrast);
}

float light_attenuation(const float density, const float cos_angle, const float beer_value)
{
    float primary = beer_law(beer_value, density);
}