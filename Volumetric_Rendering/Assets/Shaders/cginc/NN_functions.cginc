// https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#:~:text=0%2C%20inf%5D.-,ReLU,)%2C%20but%20with%20better%20performance.
// float sigmoid(float x)
// {
//     return 1.0f / (1.0f + exp(-x));
// }
//
// float relu(float x)
// {
//     return max(0.0f, x);
// }
//
// float sample_density_from_nn(float3 position, float weights[3][3])
// {
//     float neuron = 0.0f;
//     for (int column = 0; column < 3; column++)
//     {
//         for (int row = 0; row < 3; row++)
//         {
//             neuron += dot(position[row], weights[row][column]);
//         }
//     }
//     return sigmoid(neuron);
// }