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
	const auto number = dis(generator);
	return number;
}

inline float remap(float& value, float oldMin, float oldMax, float newMin, float newMax)
{
	return newMin + (((value - oldMin) / (oldMax - oldMin)) * (newMax - newMin));
}

inline float relu(const float x) { return std::max(0.0f, x); }
inline float sigmoid(const float x) {return 1.0f / (1.0f + exp(-x));}

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

float sampleDensity(const Vector3& samplePosition, const std::vector<float>& weights, const std::vector<float>& biases, const float& bias)
{
	// Input -> hidden layer 1
	const float n_1 = relu(weights[0] * samplePosition.x + weights[1] * samplePosition.y + weights[2] * samplePosition.z + biases[0]);
	const float n_2 = relu(weights[3] * samplePosition.x + weights[4] * samplePosition.y + weights[5] * samplePosition.z + biases[1]);
	const float n_3 = relu(weights[6] * samplePosition.x + weights[7] * samplePosition.y + weights[8] * samplePosition.z + biases[2]);

	// Hidden layer 1 -> hidden layer 2
	const float n_4 = relu(weights[9] * n_1 + weights[10] * n_2 + weights[11] * n_3 + biases[3]);
	const float n_5 = relu(weights[12] * n_1 + weights[13] * n_2 + weights[14] * n_3 + biases[4]);
	const float n_6 = relu(weights[15] * n_1 + weights[16] * n_2 + weights[17] * n_3 + biases[5]);
										
	// Hidden layer 2 -> hidden layer 3
	const float n_7 = relu(weights[18] * n_4 + weights[19] * n_5 + weights[20] * n_6 + biases[6]);
	const float n_8 = relu(weights[21] * n_4 + weights[22] * n_5 + weights[23] * n_6 + biases[7]);
	const float n_9 = relu(weights[24] * n_4 + weights[25] * n_5 + weights[26] * n_6 + biases[8]);
	
	// Hidden layer 3 -> output
	return sigmoid(weights[27] * n_7 + weights[28] * n_8 + weights[29] * n_9 + biases[9]);
}

void readFromFile(const std::string& filePath, std::vector<float>& v)
{
	std::ifstream file;
	file.open(filePath);

	if (file.is_open())
	{
		std::string line;
		while (std::getline(file, line))
		{
			v.emplace_back(std::stof(line));
		}
	}
	file.close();
}

int main()
{
	std::vector<Vector3> weigths;
	std::vector<Vector3> biases;
	float bias = 0.0f;

	std::vector<float> weightsList;
	std::vector<float> biasList;

	std::string weightsFilePath = "New Folder\\Weights.txt";
	std::string biasFilePath = "New Folder\\Bias.txt";

	readFromFile(weightsFilePath, weightsList);
	readFromFile(biasFilePath, biasList);


	float mediumValue = 0.0f;

	for (size_t i = 0; i < 100; i++)
	{
		Vector3 samplePosition = Vector3::random(); // {0.5f, 0.25f, -0.5f}
		const float density = sampleDensity(samplePosition, weightsList, biasList, bias);
		mediumValue += density;
		std::cout << "Sample position: " << samplePosition << "\t\t Density: " << std::to_string(density) << std::endl;
	}

	mediumValue /= 200;
	std::cout << "Medium value: " << std::to_string(mediumValue) << std::endl;

	return 0;
}
