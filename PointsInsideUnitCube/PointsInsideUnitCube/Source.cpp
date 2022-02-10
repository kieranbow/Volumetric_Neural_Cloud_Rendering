//-----------------------------------------------------------------
// This file will generate a set of points within a unit cube

// C++ includes
#include <vector>
#include <iostream>
#include <random>
#include <functional>

// Assimp
#include <assimp\Importer.hpp>
#include <assimp\scene.h>
#include <assimp\postprocess.h>
#include <assimp\matrix4x4.h>
#include <assimp\cimport.h>

//#define DEBUG
#define MESHDATA

// Defines
constexpr float epsilon = 0.000001f;
constexpr int max_num_rays = 10;

using Indices = unsigned short;

// Basic Vector3 struct
struct Vector3
{
	Vector3() {}
	Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

	Vector3 operator+(const Vector3& a)
	{
		Vector3 vector3;
		vector3.x = this->x + a.x;
		vector3.y = this->y + a.y;
		vector3.z = this->z + a.z;
		return vector3;
	}
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

	static Vector3 cross(const Vector3& a, const Vector3& b)
	{
		float x = (a.y * b.z) - (a.z * b.y);
		float y = (a.z * b.x) - (a.x * b.z);
		float z = (a.x * b.y) - (a.y * b.x);
		return Vector3(x, y, z);
	}
	static float dot(const Vector3& a, const Vector3& b)
	{
		return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
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
	float t = 0.0f; // Ray origin to point
	int hit = 0;
	std::string name = "";
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

// Moller & Trumbore Ray-triangle intersection https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
bool intersect_triangle(Ray ray, Vector3 vert0, Vector3 vert1, Vector3 vert2, float& t, float& u, float& v)
{
	// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection

	// Find the two edge vectors
	Vector3 v0v1 = vert1 - vert0;
	Vector3 v0v2 = vert2 - vert0;

	// Begin calculating determinant - also used to calculate U parameter
	Vector3 pvec = Vector3::cross(ray.direction, v0v2);

	float det = Vector3::dot(pvec, v0v1); //dot_product(pvec, v0v1)

	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	//if (det < EPSILON) return false; // Culls backface triangles

	if (fabs(det) < epsilon) return false; // Allows Backface triangles

	float invDet = 1.0f / det;

	// Calculate distance from vert0 to ray's origin
	Vector3 tvec = ray.origin - vert0;

	// Calculate U parameter and test bounds
	u = Vector3::dot(tvec, pvec) * invDet;
	if (u < 0.0f || u > 1.0f) return false;

	// Prepare to test V parameter
	Vector3 qvec = Vector3::cross(tvec, v0v1);

	// Calculate V parameter and test bounds
	v = Vector3::dot(ray.direction, qvec) * invDet;
	if (v < 0.0f || u + v > 1.0f) return false; // u + v should never be greater than 1

	// Calculate t, ray intersection triangle
	t = Vector3::dot(v0v2, qvec) * invDet;

	return true;
}

// Generates a floating point number from min and max
float uniform_real_distribution(float min, float max)
{
	std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<float> dis(min, max);
	auto number = dis(generator);
	return number;
}

// Generate a random position between min and max
Vector3 generate_random_position(float min, float max)
{
	Vector3 position;
	position.x = uniform_real_distribution(min, max);
	position.y = uniform_real_distribution(min, max);
	position.z = uniform_real_distribution(min, max);
	return position;
}

// Generate a random direction between min and max
Vector3 generate_random_direction(float min, float max)
{
	Vector3 direction;
	direction.x = uniform_real_distribution(min, max);
	direction.y = uniform_real_distribution(min, max);
	direction.z = uniform_real_distribution(min, max);
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
		#ifdef MESHDATA
			std::cout << "Mesh from (" + file_path + ") successfully loaded\n";
			std::cout << "Mesh data:\n";
			std::cout << "Vertex count: " + std::to_string(vertices.size()) + "\n";
			std::cout << "Indices count: " + std::to_string(indices.size()) + "\n";
			std::cout << "Triangle count: " + std::to_string(indices.size() / 3) + "\n\n"; // 3 indices make a triangle
		#endif // MESHDATA


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

	std::vector<Ray> rays;

	// Generate a set of rays at each axis with different origins
	for (int i = 0; i < max_num_rays; i++) // x axis rays
	{
		Ray ray;
		ray.origin = {-2.0f, uniform_real_distribution(-0.5f, 0.5f), uniform_real_distribution(-0.5f, 0.5f) };
		ray.direction = { 1.0f, 0.0f, 0.0f };
		ray.name = "x-axis";
		rays.push_back(ray);
	}
	for (int i = 0; i < max_num_rays; i++) // y axis rays
	{
		Ray ray;
		ray.origin = { uniform_real_distribution(-0.5f, 0.5f) , -2.0f,  uniform_real_distribution(-0.5f, 0.5f) };
		ray.direction = { 0.0f, 1.0f, 0.0f };
		ray.name = "y-axis";
		rays.push_back(ray);
	}
	for (int i = 0; i < max_num_rays; i++) // z axis rays
	{
		Ray ray;
		ray.origin = { uniform_real_distribution(-0.5f, 0.5f) , uniform_real_distribution(-0.5f, 0.5f),  -2.0f };
		ray.direction = { 0.0f, 0.0f, 1.0f };
		ray.name = "z-axis";
		rays.push_back(ray);
	}

	// Loop through all triangles within the mesh and test if the rays has hit it
	for (int i = 0; i < indices.size(); i += 3)
	{
		int vertex_idx_1 = indices.at(i);
		int vertex_idx_2 = indices.at(i + 1);
		int vertex_idx_3 = indices.at(i + 2);

		// Get the three vertices of a triangle
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
			if (intersect_triangle(rays.at(j), vert0, vert1, vert2, t, u, v))
			{
				rays.at(j).hit += 1;

				#ifdef DEBUG
					std::cout << rays.at(j).name + " Ray[" + std::to_string(j) + "] hit Triangle" + "\t [" + std::to_string(idx) + "]\n";
				#endif // DEBUG
			}
			else
			{
				#ifdef DEBUG
					std::cout << rays.at(j).name + " Ray[" + std::to_string(j) + "] missed Triangle" +  "\t [" + std::to_string(idx) + "]\n";
				#endif // DEBUG
			}
		}
	}

	int hit = 0;

	// Loop through all rays and check if the number of hits made is equal to 2
	// If the rays hit an even number of triangles. Mesh is water tight
	// If the rays hit an odd number of triangles. Mesh is not water tight
	for (auto& ray : rays)
	{
		if (ray.hit % 2 == 0) // is even
		{
			#ifdef DEBUG
				std::cout << ray.name + " hit: " + std::to_string(ray.hit) + "\n";
			#endif // _DEBUG

			hit++;
		}
		else // is odd
		{
			#ifdef DEBUG
				std::cout << ray.name + " hit: " + std::to_string(ray.hit) + "\n";
			#endif // DEBUG
		}
		if (hit == rays.size())
		{
			std::cout << "Mesh is water tight\n";
		}
	}
	return 1;
}
