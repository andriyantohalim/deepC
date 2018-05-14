#include "statistics.h"

float mean_Vec_wCPU(Vector *Vec_In)
{
	float tempsum = 0;
    
    for (int i = 0; i < Vec_In->len; i++)
    {
		tempsum += Vec_In->vals[i];
    }
    
    return tempsum/Vec_In->len;
}

float mean_Mat_wCPU(Matrix *Mat_In)
{
    float tempsum = 0;
    
    for (int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempsum += Mat_In->vals[i][j];
        }
    }
    
    return tempsum/(Mat_In->row * Mat_In->col);
}

float mean_Tsr_wCPU(Tensor *Tsr_In)
{
	float tempsum = 0;
    
    for (int k = 0; k < Tsr_In->depth; k++)
    {
		for (int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				tempsum += Tsr_In->vals[k][i][j];
			}
		}
    }
    
    return tempsum/(Tsr_In->row * Tsr_In->col * Tsr_In->depth);
}

float variance_Vec_wCPU(Vector *Vec_In)
{
    float tempsumofsqrdiff = 0;
    float tempmean = 0;
    float tempVecsize = Vec_In->len;
    
    tempmean = mean_Vec_wCPU(Vec_In);
    
    for(int i = 0; i < Vec_In->len; i++)
    {
		tempsumofsqrdiff += pow((Vec_In->vals[i] - tempmean), 2);
    }
    
    return tempsumofsqrdiff/(tempVecsize - 1);	
}

float variance_Mat_wCPU(Matrix *Mat_In)
{
    float tempsumofsqrdiff = 0;
    float tempmean = 0;
    float tempMatsize = Mat_In->row * Mat_In->col;
    
    tempmean = mean_Mat_wCPU(Mat_In);
    
    for(int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempsumofsqrdiff += pow((Mat_In->vals[i][j] - tempmean), 2);
        }
    }
    
    return tempsumofsqrdiff/(tempMatsize - 1);
}

float variance_Tsr_wCPU(Tensor *Tsr_In)
{
	float tempsumofsqrdiff = 0;
    float tempmean = 0;
    float tempTsrsize = Tsr_In->row * Tsr_In->col * Tsr_In->depth;
    
    tempmean = mean_Tsr_wCPU(Tsr_In);
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for(int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				tempsumofsqrdiff += pow((Tsr_In->vals[k][i][j] - tempmean), 2);
			}
		}
	}
    
    return tempsumofsqrdiff/(tempTsrsize - 1);
}

float std_dev_Vec_wCPU(Vector *Vec_In)
{
	float tempsumofsqrdiff = 0;
    float tempmean = 0;
    
    float tempMatsize = Vec_In->len;
    
    tempmean = mean_Vec_wCPU(Vec_In);
    
	for(int i = 0; i < Vec_In->len; i++)
    {
		tempsumofsqrdiff += pow((Vec_In->vals[i] - tempmean), 2);
    }
    
    return sqrt(tempsumofsqrdiff/((tempMatsize) - 1));
}

float std_dev_Mat_wCPU(Matrix *Mat_In)
{
    float tempsumofsqrdiff = 0;
    float tempmean = 0;
    
    float tempMatsize = Mat_In->row * Mat_In->col;
    
    tempmean = mean_Mat_wCPU(Mat_In);
    
    for(int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempsumofsqrdiff += pow((Mat_In->vals[i][j] - tempmean), 2);
        }
    }
    
    return sqrt(tempsumofsqrdiff/((tempMatsize) - 1));
}

float std_dev_Tsr_wCPU(Tensor *Tsr_In)
{
	float tempsumofsqrdiff = 0;
    float tempmean = 0;
    
    float tempTsrsize = Tsr_In->row * Tsr_In->col * Tsr_In->depth;
    
    tempmean = mean_Tsr_wCPU(Tsr_In);
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for(int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				tempsumofsqrdiff += pow((Tsr_In->vals[k][i][j] - tempmean), 2);
			}
		}
    }
    
    return sqrt(tempsumofsqrdiff/((tempTsrsize) - 1));
}

Vector normalization_Vec_wCPU(Vector *Vec_In)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	float tempmean = 0;
    float tempvariance = 0;
    
    tempmean = mean_Vec_wCPU(Vec_In);
    tempvariance = variance_Vec_wCPU(Vec_In);
    
	for(int i = 0; i < Vec_In->len; i++)
	{
		tempVec.vals[i] = (Vec_In->vals[i] - tempmean)/sqrt(tempvariance + 0.0000001f);
	}

    return tempVec;
}

Matrix normalization_Mat_wCPU(Matrix *Mat_In)
{
    Matrix tempMat = create_matrix(Mat_In->row, Mat_In->col);
    
    float tempmean = 0;
    float tempvariance = 0;
    
    tempmean = mean_Mat_wCPU(Mat_In);
    tempvariance = variance_Mat_wCPU(Mat_In);
    
    for(int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[i][j] = (Mat_In->vals[i][j] - tempmean)/sqrt(tempvariance + 0.0000001f);
        }
    }
    
    return tempMat;
}

Tensor normalization_Tsr_wCPU(Tensor *Tsr_In)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);
    
    float tempmean = 0;
    float tempvariance = 0;
    
    tempmean = mean_Tsr_wCPU(Tsr_In);
    tempvariance = variance_Tsr_wCPU(Tsr_In);
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for(int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[k][i][j] = (Tsr_In->vals[k][i][j] - tempmean)/sqrt(tempvariance + 0.0000001f);
			}
		}
    }
    
    return tempTsr;
}
