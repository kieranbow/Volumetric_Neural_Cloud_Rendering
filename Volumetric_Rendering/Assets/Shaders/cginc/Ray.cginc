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
// Function from: http://www.cse.chalmers.se/~uffe/xjobb/RurikH%C3%B6gfeldt.pdf, Algorithm 5.2
float3 ray_sphere(const float3 pos, const float radius, const ray ray, const float3 cam_pos)
{
	const float3 a = dot(ray.direction, ray.direction) * 2.0f;
	const float3 b = dot(ray.direction, ray.origin) * 2.0f;
	const float3 c = dot(ray.origin, pos);

	const float3 discriminant = b * b - 2.0f * a * (c - radius * radius);
	const float t = max((-b + sqrt(discriminant)) / a, 0.0f);
	float3 intersect = cam_pos + ray.direction * t;

	return intersect;
}