// C++ includes
#include <vector>
#include <iostream>

// Assimp
#include <assimp\Importer.hpp>
#include <assimp\scene.h>
#include <assimp\postprocess.h>
#include <assimp\matrix4x4.h>
#include <assimp\cimport.h>

struct Vector3
{
	Vector3() {}
	Vector3(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
	int x = 0;
	int y = 0;
	int z = 0;
};

struct Vertex
{
	Vertex() {}
	Vertex(int x, int y, int z) : position(x, y, z) {}
	Vector3 position = { 0, 0, 0 };
};


Assimp::Importer importer;
const aiScene* pScene;
const aiMesh* pMesh;

std::vector<Vertex> vertices;
std::vector<unsigned short> indices;

bool LoadMeshData(std::vector<Vertex>& vertices, std::vector<unsigned short>& indices)
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

			vertices.push_back(vertex);
		}

		for (unsigned int i = 0; i < pMesh->mNumFaces; i++)
		{
			aiFace* face = &pMesh->mFaces[i];

			if (face->mNumIndices == 3)
			{
				indices.push_back(face->mIndices[0]);
				indices.push_back(face->mIndices[1]);
				indices.push_back(face->mIndices[2]);
			}
		}
	}
	return true;
}

int main()
{
	// This file will generate a set of points within a unit cube

	pScene = importer.ReadFile("Something here", aiProcess_JoinIdenticalVertices | aiProcess_Triangulate);

	if (pScene == NULL)
	{
		std::cout << "Pointer to scene is null";
		return 0;
	}

	if (!LoadMeshData(vertices, indices))
	{
		std::cout << "Mesh could not be loaded";
		return 0;
	}

	return 0;
}