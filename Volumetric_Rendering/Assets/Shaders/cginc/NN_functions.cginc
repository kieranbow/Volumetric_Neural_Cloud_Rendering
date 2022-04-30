// https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#:~:text=0%2C%20inf%5D.-,ReLU,)%2C%20but%20with%20better%20performance.
float sigmoid(const float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float relu(const float x)
{
    return max(0.0f, x);
}

float nn_dot(const float inputs, const float weight/*, const float bias*/)
{
    return weight * inputs; /* + bias*/;
}

void calculate_layer(float inputs[3], inout float node[3], float weights[3][3], float bias[3])
{
    for (int n = 0; n < 3; n++)
    {
        for (int column = 0; column < 3; column++)
        {
            for (int row = 0; row < 3; row++)
            {
                node[n] += nn_dot(inputs[row], weights[row][column]) + bias[row];
            }
        }
        node[n] = relu(node[n]);
    }
}

float calculate_output(float inputs[3], float weights[3], float bias)
{
    float output = 0.0f;
    for (int row = 0; row < 3; row++)
    {
        output += nn_dot(inputs[row], weights[row]) + bias;
    }
    return relu(output);
}
