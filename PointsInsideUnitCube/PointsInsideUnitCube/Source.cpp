int main()
{
	// This file will generate a set of points within a unit cube

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
		Vertex(int x, int y, int z) : position(x, y, z){}
		Vector3 position = { 0, 0, 0 };
	};


	return 0;
}