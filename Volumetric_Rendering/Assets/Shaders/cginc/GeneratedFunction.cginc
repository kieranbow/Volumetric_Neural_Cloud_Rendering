float relu(const float x)
{ 
	return max(0.0f, x);
}
float sigmoid(const float x)
{ 
	return 1.0f / (1.0f + exp(-x));
}
float calculateDensityFromANN(float input[6], float weights[114], float bias[19])
{
	const float n1 = relu(weights[0] * input[0] + weights[1] * input[1] + weights[2] * input[2] + weights[3] * input[3] + weights[4] * input[4] + weights[5] * input[5] + bias[0]);
	const float n2 = relu(weights[6] * input[0] + weights[7] * input[1] + weights[8] * input[2] + weights[9] * input[3] + weights[10] * input[4] + weights[11] * input[5] + bias[1]);
	const float n3 = relu(weights[12] * input[0] + weights[13] * input[1] + weights[14] * input[2] + weights[15] * input[3] + weights[16] * input[4] + weights[17] * input[5] + bias[2]);
	const float n4 = relu(weights[18] * input[0] + weights[19] * input[1] + weights[20] * input[2] + weights[21] * input[3] + weights[22] * input[4] + weights[23] * input[5] + bias[3]);
	const float n5 = relu(weights[24] * input[0] + weights[25] * input[1] + weights[26] * input[2] + weights[27] * input[3] + weights[28] * input[4] + weights[29] * input[5] + bias[4]);
	const float n6 = relu(weights[30] * input[0] + weights[31] * input[1] + weights[32] * input[2] + weights[33] * input[3] + weights[34] * input[4] + weights[35] * input[5] + bias[5]);

	const float n7 = relu(weights[36] * n1 + weights[37] * n2 + weights[38] * n3 + weights[39] * n4 + weights[40] * n5 + weights[41] * n6 + bias[6]);
	const float n8 = relu(weights[42] * n1 + weights[43] * n2 + weights[44] * n3 + weights[45] * n4 + weights[46] * n5 + weights[47] * n6 + bias[7]);
	const float n9 = relu(weights[48] * n1 + weights[49] * n2 + weights[50] * n3 + weights[51] * n4 + weights[52] * n5 + weights[53] * n6 + bias[8]);
	const float n10 = relu(weights[54] * n1 + weights[55] * n2 + weights[56] * n3 + weights[57] * n4 + weights[58] * n5 + weights[59] * n6 + bias[9]);
	const float n11 = relu(weights[60] * n1 + weights[61] * n2 + weights[62] * n3 + weights[63] * n4 + weights[64] * n5 + weights[65] * n6 + bias[10]);
	const float n12 = relu(weights[66] * n1 + weights[67] * n2 + weights[68] * n3 + weights[69] * n4 + weights[70] * n5 + weights[71] * n6 + bias[11]);

	const float n13 = relu(weights[72] * n7 + weights[73] * n8 + weights[74] * n9 + weights[75] * n10 + weights[76] * n11 + weights[77] * n12 + bias[12]);
	const float n14 = relu(weights[78] * n7 + weights[79] * n8 + weights[80] * n9 + weights[81] * n10 + weights[82] * n11 + weights[83] * n12 + bias[13]);
	const float n15 = relu(weights[84] * n7 + weights[85] * n8 + weights[86] * n9 + weights[87] * n10 + weights[88] * n11 + weights[89] * n12 + bias[14]);
	const float n16 = relu(weights[90] * n7 + weights[91] * n8 + weights[92] * n9 + weights[93] * n10 + weights[94] * n11 + weights[95] * n12 + bias[15]);
	const float n17 = relu(weights[96] * n7 + weights[97] * n8 + weights[98] * n9 + weights[99] * n10 + weights[100] * n11 + weights[101] * n12 + bias[16]);
	const float n18 = relu(weights[102] * n7 + weights[103] * n8 + weights[104] * n9 + weights[105] * n10 + weights[106] * n11 + weights[107] * n12 + bias[17]);

	return sigmoid(weights[108] * n13 + weights[109] * n14 + weights[109] * n15 + weights[110] * n16 + weights[111] * n17 + weights[112] * n18 + bias[18]);
}