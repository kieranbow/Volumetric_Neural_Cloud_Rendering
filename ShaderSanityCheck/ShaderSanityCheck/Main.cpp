#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

// Random floats
inline float uniform_real_distribution()
{
	std::mt19937 generator(std::random_device{}());
	std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
	auto number = dis(generator);
	return number;
}

inline float remap(float& value, float oldMin, float oldMax, float newMin, float newMax)
{
	return newMin + (((value - oldMin) / (oldMax - oldMin)) * (newMax - newMin));
}

inline float relu(float x) { return std::max(0.0f, x); }

// Vector3 struct
struct Vector3
{
	Vector3() {}
	Vector3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {};

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

	friend std::ostream& operator<<(std::ostream& _lhs, const Vector3& _rhs)
	{
		_lhs << "x: " + std::to_string(_rhs.x) << ", " << "y: " + std::to_string(_rhs.y) << ", " << "z: " + std::to_string(_rhs.z);
		return _lhs;
	}

	static inline float dot_product(Vector3& a, Vector3& b){ return (a.x * b.x) + (a.y * b.y) + (a.z * b.z); }

	static inline Vector3 cross_product(Vector3& a, Vector3& b)
	{
		float x = (a.y * b.z) - (a.z * b.y);
		float y = (a.z * b.x) - (a.x * b.z);
		float z = (a.x * b.y) - (a.y * b.x);
		return Vector3(x, y, z);
	}

	static inline Vector3 random(){ return Vector3(uniform_real_distribution(), uniform_real_distribution(), uniform_real_distribution()); }

	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
};

float sampleDensity(const Vector3& samplePosition, const std::vector<Vector3>& weights, const std::vector<Vector3>& biases, const float& bias)
{
	// Layer 1 - Input -> hidden layer 1
	float n_1 = relu(weights[0].x * samplePosition.x + weights[0].y * samplePosition.y + weights[0].z * samplePosition.z + biases[0].x);
	float n_2 = relu(weights[1].x * samplePosition.x + weights[1].y * samplePosition.y + weights[1].z * samplePosition.z + biases[0].y);
	float n_3 = relu(weights[2].x * samplePosition.x + weights[2].y * samplePosition.y + weights[2].z * samplePosition.z + biases[0].z);

	// Layer 2 - Hidden layer 1 -> hidden layer 2
	float n_4 = relu(weights[3].x * n_1 + weights[3].y * n_2 + weights[3].z * n_3 + biases[1].x);
	float n_5 = relu(weights[4].x * n_1 + weights[4].y * n_2 + weights[4].z * n_3 + biases[1].y);
	float n_6 = relu(weights[5].x * n_1 + weights[5].y * n_2 + weights[5].z * n_3 + biases[1].z);
										
	// Layer 3 - Hidden layer 2 -> hidden layer 3
	float n_7 = relu(weights[6].x * n_4 + weights[6].y * n_5 + weights[6].z * n_6 + biases[2].x);
	float n_8 = relu(weights[7].x * n_4 + weights[7].y * n_5 + weights[7].z * n_6 + biases[2].y);
	float n_9 = relu(weights[8].x * n_4 + weights[8].y * n_5 + weights[8].z * n_6 + biases[2].z);
	
	// Layer 4 - Hidden layer 3 -> output
	float n_10 = relu(weights[9].x * n_7 + weights[9].y * n_8 + weights[9].z * n_9 + bias);

	return n_10;
}

int main()
{
	std::vector<Vector3> weigths;
	std::vector<Vector3> biases;
	float bias = 0.0f;

	std::ifstream file;
	file.open("Weights.txt");

	if (file.is_open())
	{
		std::vector<float> values;

		std::string line = "";
		while (std::getline(file, line))
		{
			values.push_back(std::stof(line));
		}

		// Layer 1
		weigths.push_back(Vector3(values[0], values[1], values[2]));
		weigths.push_back(Vector3(values[3], values[4], values[5]));
		weigths.push_back(Vector3(values[6], values[7], values[8]));
		biases.push_back(Vector3(values[9], values[10], values[11]));

		// Layer 2
		weigths.push_back(Vector3(values[12], values[13], values[14]));
		weigths.push_back(Vector3(values[15], values[16], values[17]));
		weigths.push_back(Vector3(values[18], values[19], values[20]));
		biases.push_back(Vector3(values[21], values[22], values[23]));

		// Layer 3
		weigths.push_back(Vector3(values[24], values[25], values[26]));
		weigths.push_back(Vector3(values[27], values[28], values[29]));
		weigths.push_back(Vector3(values[30], values[31], values[32]));
		biases.push_back(Vector3(values[33], values[34], values[35]));

		// Output
		weigths.push_back(Vector3(values[36], values[37], values[38]));
		bias = values[39];

		values.clear();
	}

	float mediumValue = 0.0f;

	for (size_t i = 0; i < 200; i++)
	{
		Vector3 samplePosition = Vector3::random();
		float density = sampleDensity(samplePosition, weigths, biases, bias);
		mediumValue += density;
		std::cout << "Sample position: " << samplePosition << "\t\t Density: " << std::to_string(density) << std::endl;
	}

	mediumValue /= 200;
	std::cout << "Medium value: " << std::to_string(mediumValue) << std::endl;

	return 0;
}
