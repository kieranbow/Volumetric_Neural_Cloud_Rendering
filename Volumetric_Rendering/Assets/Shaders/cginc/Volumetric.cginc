// -----------------------------------------------
// Helper Function
static float pi = 3.14159265359f;

struct volume
{
    float amplitude;
    float frequency;
    float scale;
    float offset;
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

float beer_law(const float density)
{
    return exp(-density);
}