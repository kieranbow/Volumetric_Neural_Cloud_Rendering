struct ray
{
	float3 origin;
	float3 direction;
};

// -----------------------------------------------
// Helper Functions

// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
float2 ray_box_dist(const float3 bound_min, const float3 bound_max, const ray ray)
{
	const float3 t0 = (bound_min - ray.origin) / ray.direction;
	const float3 t1 = (bound_max - ray.origin) / ray.direction;

	const float3 t_min = min(t0, t1);
	const float3 t_max = max(t0, t1);

	const float dist_a = max(max(t_min.x, t_min.y), t_min.z);
	const float dist_b = min(t_max.x, min(t_max.y, t_max.z));

	const float dist_to_box = max(0, dist_a);
	const float dist_inside_box = max(0, dist_b - dist_to_box);

	return float2(dist_to_box, dist_inside_box);
}