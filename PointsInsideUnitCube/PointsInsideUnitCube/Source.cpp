//-----------------------------------------------------------------
// This file will generate a set of points within a unit cube sized mesh

// C++ includes
#include <vector>
#include <iostream>
#include <random>
#include <functional>
#include <fstream>
#include <array>
#include <time.h>

// Assimp
#include <assimp\Importer.hpp>
#include <assimp\scene.h>
#include <assimp\postprocess.h>
#include <assimp\matrix4x4.h>
#include <assimp\cimport.h>

// Uncomment these for extra information during execution
// #define DEBUG
#define MESHDATA
//#define VERTDATA

// Defines
constexpr float epsilon = 0.000001f;
constexpr float range_min = -0.9f;
constexpr float range_max = 0.9f;
constexpr int max_num_point = 10000;

using Indices = unsigned short;

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

	// Produces a new vector C using vector A and B
	static Vector3 cross(const Vector3& a, const Vector3& b)
	{
		float x = (a.y * b.z) - (a.z * b.y);
		float y = (a.z * b.x) - (a.x * b.z);
		float z = (a.x * b.y) - (a.y * b.x);
		return Vector3(x, y, z);
	}
	
	// Calculates the relationship between two vectors. I.e opposite vectors = -1
	static float dot(const Vector3& a, const Vector3& b)
	{
		return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
	}
	
	// Finds the magnitude from the vectors x,y,z coords
	static float magnitude(const Vector3& a)
	{
		return sqrt((a.x * a.x) + (a.y * a.y) + (a.z * a.z));
	}
	
	// Normalizes a vector using it magnitude
	static Vector3 normalize(const Vector3& a)
	{
		float mag = magnitude(a);

		if (mag < epsilon) return Vector3(0.0f, 0.0f, 0.0f);
		return Vector3(a.x / mag, a.y / mag, a.z / mag);
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

struct Triangle
{
	Triangle() {}

	Vertex vert0;
	Vertex vert1;
	Vertex vert2;
};

struct Ray
{
	Ray() {}
	Ray(Vector3 _origin, Vector3 _direction) : origin(_origin), direction(_direction) {}

	Vector3 origin = { 0.0f, 0.0f, 0.0f };
	Vector3 direction = { 0.0f, 0.0f, 0.0f };
	float t1 = 0.0f; // Ray origin to point
	int hit = 0;
	std::string name = "";
};

struct Point
{
	Point() {}
	Point(float x, float y, float z, float d) : position(x, y, z), density(d) {}
	
	Vector3 position = {0.0f, 0.0f, 0.0f};
	float density = 0.0f;
};

Assimp::Importer importer;
const aiScene* pScene;
const aiMesh* pMesh;

std::vector<Vertex> vertices;
std::vector<Indices> indices;

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

// Loads a mesh using Assimp library
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

// Moller & Trumbore Ray-triangle intersection 
bool intersect_triangle(Ray ray, Vector3 vert0, Vector3 vert1, Vector3 vert2, float& t, float& u, float& v, bool cull)
{
	// https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
	// https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection

	// Find the two edge vectors
	Vector3 v0v1 = vert1 - vert0;
	Vector3 v0v2 = vert2 - vert0;

	// Begin calculating determinant - also used to calculate U parameter
	Vector3 pvec = Vector3::cross(ray.direction, v0v2);

	float det = Vector3::dot(pvec, v0v1);

	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (cull)
	{
		if (det < epsilon) return false; // Culls backface triangles
	}
	else
	{
		if (det > -epsilon && det < epsilon) return false; // Allows Backface triangles
	}

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

// Generate a random position between a min and max Vector3
Vector3 generate_random_position(Vector3 min, Vector3 max)
{
	Vector3 position;
	position.x = uniform_real_distribution(min.x, max.x);
	position.y = uniform_real_distribution(min.y, max.y);
	position.z = uniform_real_distribution(min.z, max.z);
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

bool writeToCSV(std::ofstream& file, std::vector<Point>& points)
{
	// Make sure the file is open before writing into it
	if (file.is_open())
	{
		file << "x, y, z, density" << std::endl;

		// Loop through all the points and write into the file
		for (auto& point : points)
		{
			std::string x = std::to_string(point.position.x);
			std::string y = std::to_string(point.position.y);
			std::string z = std::to_string(point.position.z);
			std::string d = std::to_string(point.density);

			file << x + ", " + y + ", " + z + ", " + d << std::endl;
		}
		file.close();
		file.flush();
		return true;
	}
	else
	{
		std::cout << "Cannot open file\n";
		return false;
	}
	return true;
}

int main()
{
	std::string file_path = "Assets\\helmet.obj";

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

#ifdef VERTDATA
			// Loops through all triangles and prints out triangles vertices position
			for (int i = 0; i < indices.size(); i += 3)
			{
				int vertex_idx_1 = indices.at(i);
				int vertex_idx_2 = indices.at(i + 1);
				int vertex_idx_3 = indices.at(i + 2);

				Vector3 vertex_1 = vertices.at(vertex_idx_1).position;
				Vector3 vertex_2 = vertices.at(vertex_idx_2).position;
				Vector3 vertex_3 = vertices.at(vertex_idx_3).position;

				std::string vertex_1_pos = std::to_string(vertex_1.x) + ", \t" + std::to_string(vertex_1.y) + ", \t" + std::to_string(vertex_1.z);
				std::string vertex_2_pos = std::to_string(vertex_2.x) + ", \t" + std::to_string(vertex_2.y) + ", \t" + std::to_string(vertex_2.z);
				std::string vertex_3_pos = std::to_string(vertex_3.x) + ", \t" + std::to_string(vertex_3.y) + ", \t" + std::to_string(vertex_3.z);

				int idx = (i / 3) + 1;
				std::cout << "Triangle[" + std::to_string(idx) + "] position: \n";
				std::cout << "Vertex 1: " + vertex_1_pos + "\n";
				std::cout << "Vertex 2: " + vertex_2_pos + "\n";
				std::cout << "Vertex 3: " + vertex_3_pos + "\n\n";
			}
#endif // VERTDATA
	}

	// Create the 3 files which will be used for training, validing and testing the AI
	// These files will have x,y,z positions stored in them
	std::vector<std::ofstream> files;
	files.push_back(std::ofstream("Training.csv"));
	files.push_back(std::ofstream("Validation.csv"));
	files.push_back(std::ofstream("Testing.csv"));

	// Loop through the 3 files
	// Generate a set of points
	// Write the points inside the files
	for (auto& file : files)
	{
		std::vector<Point> rand_points;
		std::vector<Point> saved_points;

		// First, generate a set of points within a unit cube space. 
		// I.e a range from [-1, 1]
		for (int i = 0; i < max_num_point; i++)
		{
			Point point;
			point.position = generate_random_position(range_min, range_max);
			point.density = 1.0f;// uniform_real_distribution(0.0f, 1.0f);
			rand_points.push_back(point);
		}

		// Next, loop through each point generating a ray and check if the point is within the mesh
		// If the point is inside the mesh, save it into a json file
		// If the point is not inside the mesh, ignore it.
		for (auto& point : rand_points)
		{
			Ray ray;
			ray.origin = point.position;
			ray.direction = Vector3(1.0f, 0.0f, 0.0f); // Direction can be randomized but for simplicity, it set on the x+ axis

			// Loop through the mesh's triangles and check if ray collided
			// using Moller Trumbore Ray Triangle Intersection
			for (int i = 0; i < indices.size(); i += 3)
			{
				float u = 0;
				float v = 0;
				float t = 0;

				int vertex_idx_1 = indices.at(i);
				int vertex_idx_2 = indices.at(i + 1);
				int vertex_idx_3 = indices.at(i + 2);

				// Get the three vertices of a triangle
				Vector3 vert0 = vertices.at(vertex_idx_1).position;
				Vector3 vert1 = vertices.at(vertex_idx_2).position;
				Vector3 vert2 = vertices.at(vertex_idx_3).position;

				// Test if ray hits the triangle
				intersect_triangle(ray, vert0, vert1, vert2, t, u, v, false);

				// Check if the t value of the ray is positive, increment the rays hit value
				if (t > 0.0f)
				{
					ray.hit++;
				}
			}

			// If the rays hit value is odd, then its inside a mesh
			// If the rays hit value is even, then its outside the mesh
			if (ray.hit & 1) // Bitwise operator
			{
				Point temp;
				temp.position = point.position;
				temp.density = point.density;
				saved_points.push_back(temp);
			}

			// Once all the correct points are store inside saved_points.
			// Clear the rand_points since it will not be used anymore.
			rand_points.clear();
		}
		// Finally, when all points are generated, save them to a json file
		std::cout << "Writing saved points into file" << std::endl;
		writeToCSV(file, saved_points);
		std::cout << "Finished writing to file. Closing file..." << std::endl;
	}
	return 1;
}
