#include "fully_connected.h"

Vector Fully_Connected_wCPU(Vector *Vec_In, Matrix *Mat_Weights, Vector *Vec_Bias)
{
	assert(Mat_Weights->col == Vec_In->len);
	assert(Mat_Weights->row == Vec_Bias->len);

	Vector tempVec = create_vector(Mat_Weights->row);
	
	for(int i = 0; i < Mat_Weights->row; i++)
	{
		float tempval = 0.0f;
		
		for(int j = 0; j < Mat_Weights->col; j++)
		{
			tempval += Mat_Weights->vals[i][j] * Vec_In->vals[j];
		}
		
		tempVec.vals[i] = tempval + Vec_Bias->vals[i];
	}
	
	return tempVec;
}
