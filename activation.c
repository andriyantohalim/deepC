#include "activation.h"

Vector ReLU_Vec_wCPU(Vector *Vec_In)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		if (Vec_In->vals[i] > 0)
		{
			tempVec.vals[i] = Vec_In->vals[i];
		}
		else
		{    
			tempVec.vals[i] = 0;
		}
	}
	
	return tempVec;
}

Matrix ReLU_Mat_wCPU(Matrix *Mat_In)
{
    Matrix tempMat = create_matrix(Mat_In->row, Mat_In->col);
        
    for (int i = 0; i < Mat_In->row; i++)
    {
        for (int j = 0; j < Mat_In->col; j++)
        {
            if (Mat_In->vals[i][j] > 0)
            {
                tempMat.vals[i][j] = Mat_In->vals[i][j];
            }
            else
            {    
                tempMat.vals[i][j] = 0;
            }
        }
    }
    
    return tempMat;
}

Tensor ReLU_Tsr_wCPU(Tensor *Tsr_In)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);
        
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for (int i = 0; i < Tsr_In->row; i++)
		{
			for (int j = 0; j < Tsr_In->col; j++)
			{
				if (Tsr_In->vals[k][i][j] > 0)
				{
					tempTsr.vals[k][i][j] = Tsr_In->vals[k][i][j];
				}
				else
				{    
					tempTsr.vals[k][i][j] = 0;
				}
			}
		}
	}
    
    return tempTsr;
}

Vector Leaky_ReLU_Vec_wCPU(Vector *Vec_In)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		if (Vec_In->vals[i] > 0)
		{
			tempVec.vals[i] = Vec_In->vals[i];
		}
		else
		{    
			tempVec.vals[i] = 0.01*Vec_In->vals[i];
		}
	}
	
	return tempVec;
}

Matrix Leaky_ReLU_Mat_wCPU(Matrix *Mat_In)
{
    Matrix tempMat = create_matrix(Mat_In->row, Mat_In->col);
    
    for (int i = 0; i < Mat_In->row; i++)
    {
        for (int j = 0; j < Mat_In->col; j++)
        {
            if (Mat_In->vals[i][j] > 0)
            {
                tempMat.vals[i][j] = Mat_In->vals[i][j];
            }
            else
            {    
                tempMat.vals[i][j] = 0.01*Mat_In->vals[i][j];
            }
        }
    }
    
    return tempMat;
}

Tensor Leaky_ReLU_Tsr_wCPU(Tensor *Tsr_In)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for (int i = 0; i < Tsr_In->row; i++)
		{
			for (int j = 0; j < Tsr_In->col; j++)
			{
				if (Tsr_In->vals[k][i][j] > 0)
				{
					tempTsr.vals[k][i][j] = Tsr_In->vals[k][i][j];
				}
				else
				{    
					tempTsr.vals[k][i][j] = 0.01*Tsr_In->vals[k][i][j];
				}
			}
		}
	}
    
    return tempTsr;
}

Vector softmax_Vec_wCPU(Vector *Vec_In)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	float temp[Vec_In->len];
    double tempsum = 0;
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		temp[i] = exp(Vec_In->vals[i]);
		tempsum += temp[i];
	}
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		tempVec.vals[i] = temp[i]/tempsum;
	}
	
	return tempVec;
}

Matrix softmax_Mat_wCPU(Matrix *Mat_In)
{
    Matrix tempMat = create_matrix(Mat_In->row, Mat_In->col);
    
    float temp[Mat_In->row][Mat_In->col];
    double tempsum = 0;
    
    for(int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            temp[i][j] = exp(Mat_In->vals[i][j]);
            tempsum += temp[i][j];
        }
    }
    
    for(int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[i][j] = temp[i][j]/tempsum;
        }
    }
    
    return tempMat;
}

Tensor softmax_Tsr_wCPU(Tensor *Tsr_In)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);
    
    float temp[Tsr_In->row][Tsr_In->col][Tsr_In->depth];
    double tempsum = 0;
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for(int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				temp[k][i][j] = exp(Tsr_In->vals[k][i][j]);
				tempsum += temp[k][i][j];
			}
		}
	}
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for(int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[k][i][j] = temp[k][i][j]/tempsum;
			}
		}
	}
    
    return tempTsr;
}

Vector tanh_Vec_wCPU(Vector *Vec_In)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		tempVec.vals[i] = 2.0/(1.0 + exp(-2 * Vec_In->vals[i])) - 1;
	}
	
	return tempVec;
}

Matrix tanh_Mat_wCPU(Matrix *Mat_In)
{
    Matrix tempMat = create_matrix(Mat_In->row, Mat_In->col);
    
    for (int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[i][j] = 2.0/(1.0 + exp(-2 * Mat_In->vals[i][j])) - 1;
        }
    }
    
    return tempMat;
}

Tensor tanh_Tsr_wCPU(Tensor *Tsr_In)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);
    
    for (int k = 0; k < Tsr_In->depth; k++)
    {
		for (int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[k][i][j] = 2.0/(1.0 + exp(-2 * Tsr_In->vals[k][i][j])) - 1;
			}
		}
	}
    
    return tempTsr;
}

Vector sigmoid_Vec_wCPU(Vector *Vec_In)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		tempVec.vals[i] = 1.0/(1.0 + exp(-Vec_In->vals[i]));
	}
	
	return tempVec;
}

Matrix sigmoid_Mat_wCPU(Matrix *Mat_In)
{
    Matrix tempMat = create_matrix(Mat_In->row, Mat_In->col);
    
    for(int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[i][j] = 1.0/(1.0 + exp(-Mat_In->vals[i][j]));
        }
    }
    
    return tempMat;
}

Tensor sigmoid_Tsr_wCPU(Tensor *Tsr_In)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for(int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[k][i][j] = 1.0/(1.0 + exp(-Tsr_In->vals[k][i][j]));
			}
		}
	}
    
    return tempTsr;
}



