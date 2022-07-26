float relu(const float x)
{ 
	return max(0.0f, x);
}
float sigmoid(const float x)
{ 
	return 1.0f / (1.0f + exp(-x));
}
float calculateDensityFromANN(float input[9], float weights[252], float bias[28])
{
	const float n1 = relu(weights[0] * input[0] + weights[1] * input[1] + weights[2] * input[2] + weights[3] * input[3] + weights[4] * input[4] + weights[5] * input[5] + weights[6] * input[6] + weights[7] * input[7] + weights[8] * input[8] + bias[0]);
	const float n2 = relu(weights[9] * input[0] + weights[10] * input[1] + weights[11] * input[2] + weights[12] * input[3] + weights[13] * input[4] + weights[14] * input[5] + weights[15] * input[6] + weights[16] * input[7] + weights[17] * input[8] + bias[1]);
	const float n3 = relu(weights[18] * input[0] + weights[19] * input[1] + weights[20] * input[2] + weights[21] * input[3] + weights[22] * input[4] + weights[23] * input[5] + weights[24] * input[6] + weights[25] * input[7] + weights[26] * input[8] + bias[2]);
	const float n4 = relu(weights[27] * input[0] + weights[28] * input[1] + weights[29] * input[2] + weights[30] * input[3] + weights[31] * input[4] + weights[32] * input[5] + weights[33] * input[6] + weights[34] * input[7] + weights[35] * input[8] + bias[3]);
	const float n5 = relu(weights[36] * input[0] + weights[37] * input[1] + weights[38] * input[2] + weights[39] * input[3] + weights[40] * input[4] + weights[41] * input[5] + weights[42] * input[6] + weights[43] * input[7] + weights[44] * input[8] + bias[4]);
	const float n6 = relu(weights[45] * input[0] + weights[46] * input[1] + weights[47] * input[2] + weights[48] * input[3] + weights[49] * input[4] + weights[50] * input[5] + weights[51] * input[6] + weights[52] * input[7] + weights[53] * input[8] + bias[5]);
	const float n7 = relu(weights[54] * input[0] + weights[55] * input[1] + weights[56] * input[2] + weights[57] * input[3] + weights[58] * input[4] + weights[59] * input[5] + weights[60] * input[6] + weights[61] * input[7] + weights[62] * input[8] + bias[6]);
	const float n8 = relu(weights[63] * input[0] + weights[64] * input[1] + weights[65] * input[2] + weights[66] * input[3] + weights[67] * input[4] + weights[68] * input[5] + weights[69] * input[6] + weights[70] * input[7] + weights[71] * input[8] + bias[7]);
	const float n9 = relu(weights[72] * input[0] + weights[73] * input[1] + weights[74] * input[2] + weights[75] * input[3] + weights[76] * input[4] + weights[77] * input[5] + weights[78] * input[6] + weights[79] * input[7] + weights[80] * input[8] + bias[8]);

	const float n10 = relu(weights[81] * n1 + weights[82] * n2 + weights[83] * n3 + weights[84] * n4 + weights[85] * n5 + weights[86] * n6 + weights[87] * n7 + weights[88] * n8 + weights[89] * n9 + bias[9]);
	const float n11 = relu(weights[90] * n1 + weights[91] * n2 + weights[92] * n3 + weights[93] * n4 + weights[94] * n5 + weights[95] * n6 + weights[96] * n7 + weights[97] * n8 + weights[98] * n9 + bias[10]);
	const float n12 = relu(weights[99] * n1 + weights[100] * n2 + weights[101] * n3 + weights[102] * n4 + weights[103] * n5 + weights[104] * n6 + weights[105] * n7 + weights[106] * n8 + weights[107] * n9 + bias[11]);
	const float n13 = relu(weights[108] * n1 + weights[109] * n2 + weights[110] * n3 + weights[111] * n4 + weights[112] * n5 + weights[113] * n6 + weights[114] * n7 + weights[115] * n8 + weights[116] * n9 + bias[12]);
	const float n14 = relu(weights[117] * n1 + weights[118] * n2 + weights[119] * n3 + weights[120] * n4 + weights[121] * n5 + weights[122] * n6 + weights[123] * n7 + weights[124] * n8 + weights[125] * n9 + bias[13]);
	const float n15 = relu(weights[126] * n1 + weights[127] * n2 + weights[128] * n3 + weights[129] * n4 + weights[130] * n5 + weights[131] * n6 + weights[132] * n7 + weights[133] * n8 + weights[134] * n9 + bias[14]);
	const float n16 = relu(weights[135] * n1 + weights[136] * n2 + weights[137] * n3 + weights[138] * n4 + weights[139] * n5 + weights[140] * n6 + weights[141] * n7 + weights[142] * n8 + weights[143] * n9 + bias[15]);
	const float n17 = relu(weights[144] * n1 + weights[145] * n2 + weights[146] * n3 + weights[147] * n4 + weights[148] * n5 + weights[149] * n6 + weights[150] * n7 + weights[151] * n8 + weights[152] * n9 + bias[16]);
	const float n18 = relu(weights[153] * n1 + weights[154] * n2 + weights[155] * n3 + weights[156] * n4 + weights[157] * n5 + weights[158] * n6 + weights[159] * n7 + weights[160] * n8 + weights[161] * n9 + bias[17]);

	const float n19 = relu(weights[162] * n10 + weights[163] * n11 + weights[164] * n12 + weights[165] * n13 + weights[166] * n14 + weights[167] * n15 + weights[168] * n16 + weights[169] * n17 + weights[170] * n18 + bias[18]);
	const float n20 = relu(weights[171] * n10 + weights[172] * n11 + weights[173] * n12 + weights[174] * n13 + weights[175] * n14 + weights[176] * n15 + weights[177] * n16 + weights[178] * n17 + weights[179] * n18 + bias[19]);
	const float n21 = relu(weights[180] * n10 + weights[181] * n11 + weights[182] * n12 + weights[183] * n13 + weights[184] * n14 + weights[185] * n15 + weights[186] * n16 + weights[187] * n17 + weights[188] * n18 + bias[20]);
	const float n22 = relu(weights[189] * n10 + weights[190] * n11 + weights[191] * n12 + weights[192] * n13 + weights[193] * n14 + weights[194] * n15 + weights[195] * n16 + weights[196] * n17 + weights[197] * n18 + bias[21]);
	const float n23 = relu(weights[198] * n10 + weights[199] * n11 + weights[200] * n12 + weights[201] * n13 + weights[202] * n14 + weights[203] * n15 + weights[204] * n16 + weights[205] * n17 + weights[206] * n18 + bias[22]);
	const float n24 = relu(weights[207] * n10 + weights[208] * n11 + weights[209] * n12 + weights[210] * n13 + weights[211] * n14 + weights[212] * n15 + weights[213] * n16 + weights[214] * n17 + weights[215] * n18 + bias[23]);
	const float n25 = relu(weights[216] * n10 + weights[217] * n11 + weights[218] * n12 + weights[219] * n13 + weights[220] * n14 + weights[221] * n15 + weights[222] * n16 + weights[223] * n17 + weights[224] * n18 + bias[24]);
	const float n26 = relu(weights[225] * n10 + weights[226] * n11 + weights[227] * n12 + weights[228] * n13 + weights[229] * n14 + weights[230] * n15 + weights[231] * n16 + weights[232] * n17 + weights[233] * n18 + bias[25]);
	const float n27 = relu(weights[234] * n10 + weights[235] * n11 + weights[236] * n12 + weights[237] * n13 + weights[238] * n14 + weights[239] * n15 + weights[240] * n16 + weights[241] * n17 + weights[242] * n18 + bias[26]);

	return sigmoid(weights[243] * n19 + weights[244] * n20 + weights[245] * n21 + weights[246] * n22 + weights[247] * n23 + weights[248] * n24 + weights[249] * n25 + weights[250] * n26 + weights[251] * n27 + bias[27]);
}