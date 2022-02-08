//-----------------------------------------------------------------
// This file will generate a set of points within a unit cube

// C++ includes
#include <vector>
#include <iostream>
#include <random>

// Assimp
#include <assimp\Importer.hpp>
#include <assimp\scene.h>
#include <assimp\postprocess.h>
#include <assimp\matrix4x4.h>
#include <assimp\cimport.h>

using Indices = unsigned short;

// Basic Vector3 struct
struct Vector3
{
	Vector3() {}
	Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

	// Adds two vectors together
	Vector3 operator+(const Vector3& a)
	{
		Vector3 vector3;
		vector3.x = this->x + a.x;
		vector3.y = this->y + a.y;
		vector3.z = this->z + a.z;
		return vector3;
	}

	// Subtracts two vector
	Vector3 operator-(const Vector3& a)
	{
		Vector3 vector3;
		vector3.x = this->x - a.x;
		vector3.y = this->y - a.y;
		vector3.z = this->z - a.z;
		return vector3;
	}

	Vector3 operator*(const Vector3& a)
	{
		Vector3 vector3;
		vector3.x = this->x * a.x;
		vector3.y = this->y * a.y;
		vector3.z = this->z * a.z;
		return vector3;
	}

	Vector3 operator/(const Vector3& a)
	{
		Vector3 vector3;
		vector3.x = this->x / a.x;
		vector3.y = this->y / a.y;
		vector3.z = this->z / a.z;
		return vector3;
	}

	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
};

struct Vertex
{
	Vertex() {}
	Vertex(float _x, float _y, float _z) : position(_x, _y, _z) {}
	Vector3 position = { 0.0f, 0.0f, 0.0f };
};

struct Ray
{
	Ray() {}
	Ray(Vector3 _origin, Vector3 _direction) : origin(_origin), direction(_direction) {}

	Vector3 origin = { 0.0f, 0.0f, 0.0f };
	Vector3 direction = { 0.0f, 0.0f, 0.0f };
	float t = 0; // Ray origin to point
};

Assimp::Importer importer;
const aiScene* pScene;
const aiMesh* pMesh;

std::vector<Vertex> vertices;
std::vector<Indices> indices;


// Add description here
// https://www.mathsisfun.com/algebra/vectors-dot-product.html
float dot_product(Vector3 a, Vector3 b)
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

// Produces a new vector C using vector A and B
// https://www.mathsisfun.com/algebra/vectors-cross-product.html
Vector3 cross_product(Vector3 a, Vector3 b)
{
	float x = (a.y * b.z) - (a.z * b.y);
	float y = (a.z * b.x) - (a.x * b.z);
	float z = (a.x * b.y) - (a.y * b.x);
	return Vector3(x, y, z);
}

Vector3 sub(Vector3 v1, Vector3 v2)
{
	Vector3 dest;
	dest.x = v1.x - v2.x;
	dest.y = v1.y - v2.y;
	dest.z = v1.z - v2.z;
	return dest;
}

bool LoadMeshData(std::vector<Vertex>& _vertices, std::vector<Indices>& _indices)
{
	for (unsigned int m = 0; m < pScene->mNumMeshes; m++)
	{
		pMesh = pScene->mMeshes[m];

		for (unsigned int vrtIdx = 0; vrtIdx < pMesh->mNumVertices; vrtIdx++)
		{
			Vertex vertex;

			vertex.position.x = pMesh->mVertices[vrtIdx].x;
			vertex.position.y = pMesh->mVertices[vrtIdx].y;
			vertex.position.z = pMesh->mVertices[vrtIdx].z;

			_vertices.push_back(vertex);
		}

		for (unsigned int i = 0; i < pMesh->mNumFaces; i++)
		{
			aiFace* face = &pMesh->mFaces[i];

			if (face->mNumIndices == 3)
			{
				_indices.push_back(face->mIndices[0]);
				_indices.push_back(face->mIndices[1]);
				_indices.push_back(face->mIndices[2]);
			}
		}
	}
	return true;
}

// https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
bool intersect_triangle(Ray ray, Vector3 vert0, Vector3 vert1, Vector3 vert2, float t, float u, float v, bool enable_cull)
{
	float epsilon = 0.000001f;
	Vector3 edge1;
	Vector3 edge2;
	Vector3 tvec;
	Vector3 pvec;
	Vector3 qvec;
	float determinant;
	float inv_determinant;

	// Find vectors for two edges that share vert0
	edge1 = sub(vert1, vert0);
	edge2 = sub(vert2, vert0);

	// Begin calculating determinant - also used to calculate U parameter
	pvec = cross_product(ray.direction, edge2);

	// if determinant is near zero, ray lies in plane of triangle
	determinant = dot_product(edge1, pvec);

	if (enable_cull)
	{
		// Test double sided triangle
		if (determinant < epsilon) return false;

		// Calculate distance from vert0 to ray origin
		tvec = sub(ray.origin, vert0);

		// Calculate U parameter and test bounds
		u = dot_product(tvec, pvec);
		if (u < 0.0f || u > determinant) return false;

		// Prepare to test v parameter
		qvec = cross_product(tvec, edge1);

		// Calculate V parameter and test bounds
		v = dot_product(ray.origin, qvec);
		if (v < 0.0f || u + v > determinant) return false; // u + v cannot be greater than 1

		// Calculate t, scale parameters, ray intersects triangle
		t = dot_product(edge2, qvec);
		inv_determinant = 1.0f / determinant;

		t *= inv_determinant;
		u *= inv_determinant;
		v *= inv_determinant;

	}
	if (!enable_cull)
	{
		if (determinant > -epsilon && determinant < epsilon) return false;

		inv_determinant = 1.0f / determinant;

		// Calculate distance from vert0 to ray origin
		tvec = sub(ray.origin, vert0);

		// Calculate U parameter and test bounds
		u = dot_product(tvec, pvec) * inv_determinant;
		if (u < 0.0f || u > 1.0f) return false;

		// Prepare to test v parameter
		qvec = cross_product(tvec, edge1);

		// Calculate V parameter and test bounds
		v = dot_product(ray.origin, qvec) * inv_determinant;
		if (v < 0.0f || u + v > 1.0f) return false; // u + v cannot be greater than 1

		// Calculate t, scale parameters, ray intersects triangle
		t = dot_product(edge2, qvec) * inv_determinant;
	}

	return true;
}

// Generate a number from min and max
int uniformDistribution(int min, int max)
{
	std::default_random_engine random_engine(static_cast<unsigned int>(std::random_device{}()));
	std::uniform_int_distribution<> dis(min, max);
	int gen_number = dis(random_engine);
	return gen_number;
}

// Generate a position using uniform Distribution between min and max
Vector3 generate_random_position(int min, int max)
{
	Vector3 position;
	position.x = static_cast<float>(uniformDistribution(min, max));
	position.y = static_cast<float>(uniformDistribution(min, max));
	position.z = static_cast<float>(uniformDistribution(min, max));
	return position;
}

// Generate a direction using uniform Distribution between min and max
Vector3 generate_random_direction(int min, int max)
{
	Vector3 direction;
	direction.x = static_cast<float>(uniformDistribution(min, max));
	direction.y = static_cast<float>(uniformDistribution(min, max));
	direction.z = static_cast<float>(uniformDistribution(min, max));
	return direction;
}

int main()
{
	std::string file_path = "Assets\\Unit_Cube.obj";

	pScene = importer.ReadFile(file_path, aiProcess_JoinIdenticalVertices | aiProcess_Triangulate);

	// Checks to see if aiScene is valid before loading the mesh
	if (pScene == NULL)
	{
		std::cout << "Pointer to scene is null\n";
		return 0;
	}

	// Checks to see if the mesh could be loaded
	if (!LoadMeshData(vertices, indices))
	{
		std::cout << "Mesh from (" + file_path + ") could not be loaded\n";
		return 0;
	}
	else
	{
		std::cout << "Mesh from (" + file_path + ") successfully loaded\n";
		std::cout << "Mesh data:\n";
		std::cout << "Vertex count: "	+ std::to_string(vertices.size()) + "\n";
		std::cout << "Indices count: "	+ std::to_string(indices.size()) + "\n";
		std::cout << "Triangle count: "	+ std::to_string(indices.size() / 3) + "\n\n"; // 3 indices make a triangle

		//for (int i = 0; i < vertices.size(); i++)
		//{
		//	float x = vertices.at(i).position.x;
		//	float y = vertices.at(i).position.y;
		//	float z = vertices.at(i).position.z;

		//	std::string position = std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z);
		//	std::cout << "Vertex[" + std::to_string(i) + "] position: " + position + "\n";
		//}

		//for (int i = 0; i < indices.size(); i += 3)
		//{
		//	int vertex_idx_1 = indices.at(i);
		//	int vertex_idx_2 = indices.at(i + 1);
		//	int vertex_idx_3 = indices.at(i + 2);

		//	Vector3 vertex_1 = vertices.at(vertex_idx_1).position;
		//	Vector3 vertex_2 = vertices.at(vertex_idx_2).position;
		//	Vector3 vertex_3 = vertices.at(vertex_idx_3).position;

		//	std::string vertex_1_pos = std::to_string(vertex_1.x) + ", \t" + std::to_string(vertex_1.y) + ", \t" + std::to_string(vertex_1.z);
		//	std::string vertex_2_pos = std::to_string(vertex_2.x) + ", \t" + std::to_string(vertex_2.y) + ", \t" + std::to_string(vertex_2.z);
		//	std::string vertex_3_pos = std::to_string(vertex_3.x) + ", \t" + std::to_string(vertex_3.y) + ", \t" + std::to_string(vertex_3.z);

		//	int idx = (i / 3) + 1;
		//	std::cout << "Triangle[" + std::to_string(idx) + "] position: \n";
		//	std::cout << "Vertex 1: " + vertex_1_pos + "\n";
		//	std::cout << "Vertex 2: " + vertex_2_pos + "\n";
		//	std::cout << "Vertex 3: " + vertex_3_pos + "\n\n";
		//}
	}

	Ray test_ray;
	test_ray.origin = { -2.0f, -0.25f, 0.45f };
	test_ray.direction = { 1.0f, 0.0f, 0.0f };

	Vector3 world_origin(0.0f, 0.0f, 0.0f);
	
	std::vector<Ray> rays;

	// Generate a set of rays with different origins and directions
	for (int i = 0; i < 10; i++)
	{
		Ray ray;
		ray.origin = generate_random_position(-1, 5);
		ray.direction = generate_random_direction(-1, 5) - world_origin;
		rays.push_back(ray);
	}

	size_t total = rays.size() * (indices.size() / 3);
	int hit = 0;

	// Loop through all triangles within the mesh and test if a ray has hit it
	for (int i = 0; i < indices.size(); i += 3)
	{
		int vertex_idx_1 = indices.at(i);
		int vertex_idx_2 = indices.at(i + 1);
		int vertex_idx_3 = indices.at(i + 2);

		Vector3 vert0 = vertices.at(vertex_idx_1).position;
		Vector3 vert1 = vertices.at(vertex_idx_2).position;
		Vector3 vert2 = vertices.at(vertex_idx_3).position;

		float t = 0.0f;
		float u = 0.0f;
		float v = 0.0f;
		
		int idx = (i / 3);

		// Loop through all rays and check if any hit the triangle.
		for (int j = 0; j < rays.size(); j++)
		{
			if (intersect_triangle(rays.at(j), vert0, vert1, vert2, t, u, v, false))
			{
				std::cout << "Ray[" + std::to_string(j) + "] hit Triangle" + "\t [" + std::to_string(idx) + "]\n";
				hit++;
			}
			else
			{
				std::cout << "Ray[" + std::to_string(j) + "] missed Triangle" +  "\t [" + std::to_string(idx) + "]\n";
			}
		}

		std::cout << "-------------------------------------\n";

		//if (intersect_triangle(test_ray, vert0, vert1, vert2, t, u, v, false))
		//{
		//	std::cout << "Ray [] Hit Triangle [" + std::to_string(idx) + "]\n";

		//	//std::string vertex_1_pos = std::to_string(vert0.x) + ", \t" + std::to_string(vert0.y) + ", \t" + std::to_string(vert0.z);
		//	//std::string vertex_2_pos = std::to_string(vert1.x) + ", \t" + std::to_string(vert1.y) + ", \t" + std::to_string(vert1.z);
		//	//std::string vertex_3_pos = std::to_string(vert2.x) + ", \t" + std::to_string(vert2.y) + ", \t" + std::to_string(vert2.z);

		//	//std::cout << "Vertex 1: " + vertex_1_pos + "\n";
		//	//std::cout << "Vertex 2: " + vertex_2_pos + "\n";
		//	//std::cout << "Vertex 3: " + vertex_3_pos + "\n";

		//}
		//else
		//{
		//	std::cout << "Miss Triangle [" + std::to_string(idx) + "]\n";
		//}
	}

	std::cout << std::to_string(hit) + "/" + std::to_string(total) + "\n";

	return 0;
}
