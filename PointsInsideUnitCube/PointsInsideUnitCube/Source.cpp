//-----------------------------------------------------------------
// This file will generate a set of points within a unit cube

// C++ includes
#include <vector>
#include <iostream>

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
	Vector3 origin = { 0.0f, 0.0f, 0.0f };
	Vector3 direction = { 0.0f, 0.0f, 0.0f };
	float t = 0;
};

Assimp::Importer importer;
const aiScene* pScene;
const aiMesh* pMesh;

std::vector<Vertex> vertices;
std::vector<Indices> indices;

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

//
float dot_product(Vector3 a, Vector3 b)
{
	// https://www.mathsisfun.com/algebra/vectors-dot-product.html
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

// Produces a new vector C using vector A and B
// // https://www.mathsisfun.com/algebra/vectors-cross-product.html
Vector3 cross_product(Vector3 a, Vector3 b)
{
	float x = a.y * b.z - a.z * b.y;
	float y = a.z * b.x - a.x * b.z;
	float z = a.x * b.y - a.y * b.x;
	return Vector3(x, y, z);
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
		std::cout << "Triangle count: "	+ std::to_string(indices.size() / 3) + "\n"; // 3 indices make a triangle

		for (int i = 0; i < vertices.size(); i++)
		{
			float x = vertices.at(i).position.x;
			float y = vertices.at(i).position.y;
			float z = vertices.at(i).position.z;

			std::string position = std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z);
			std::cout << "Vertex[" + std::to_string(i) + "] position: " + position + "\n";
		}
	}

	Vector3 a = { 2.0f, 3.0f, 4.0f };
	Vector3 b = { 5.0f, 6.0f, 7.0f };

	float dot = dot_product(a, b);
	std::cout << std::to_string(dot) + "\n";

	Vector3 cross = cross_product(a, b);
	std::string cross_x = std::to_string(cross.x);
	std::string cross_y = std::to_string(cross.y);
	std::string cross_z = std::to_string(cross.z);
	std::cout << cross_x + ", " + cross_y + ", " + cross_z + "\n";

	return 0;
}
