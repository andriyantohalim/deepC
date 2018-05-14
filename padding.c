#include "padding.h"

Vector padding_asymmetric_Vec_wCPU(Vector *Vec_In, int padsize)
{
	assert(padsize > 0);
	
	int templen = Vec_In->len + padsize;
	
	Vector tempVec = create_vector(templen);
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		tempVec.vals[i] = Vec_In->vals[i];
	}
	
	return tempVec;
}

Matrix padding_2d_asymmetric_Mat_wCPU(Matrix *Mat_In, int row_padsize, int col_padsize)
{
    assert(row_padsize > 0);
	assert(col_padsize > 0);
	
	int temprow = Mat_In->row + row_padsize;
    int tempcol = Mat_In->col + col_padsize;
    
    Matrix tempMat = create_matrix(temprow, tempcol);
    
    for (int i = 0; i < Mat_In->row; i++)
    {
        for (int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[i][j] = Mat_In->vals[i][j];
        }
    }
    
    return tempMat;
}

Tensor padding_2d_asymmetric_Tsr_wCPU(Tensor *Tsr_In, int row_padsize, int col_padsize)
{
	assert(row_padsize > 0);
	assert(col_padsize > 0);
	
	int temprow = Tsr_In->row + row_padsize;
    int tempcol = Tsr_In->col + col_padsize;
    
    Tensor tempTsr = create_tensor(temprow, tempcol, Tsr_In->depth);
    
    for (int k = 0; k < Tsr_In->depth; k++)
    {
		for (int i = 0; i < Tsr_In->row; i++)
		{
			for (int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[i][j] = Tsr_In->vals[i][j];
			}
		}
    }
    
    return tempTsr;
}

void vpadding_asymmetric_Vec_wCPU(Vector *Vec_In, int padsize)
{
	assert(padsize > 0);
	
	Vec_In->len += padsize;
	
	Vec_In->vals = realloc(Vec_In->vals, Vec_In->len * sizeof(float));
}

void vpadding_2d_asymmetric_Mat_wCPU(Matrix *Mat_In, int row_padsize, int col_padsize)
{
	assert(row_padsize > 0);
	assert(col_padsize > 0);
	
	Mat_In->row += row_padsize;
	Mat_In->col += col_padsize;
	
	// create bigger matrix based on the row- and col- padsize
	Mat_In->vals = realloc(Mat_In->vals, Mat_In->row * sizeof(float *));
	
	for(int i = 0; i < Mat_In->row; i++)
	{
		Mat_In->vals[i] = realloc(Mat_In->vals[i], Mat_In->col * sizeof(float));
	}
}

void vpadding_2d_asymmetric_Tsr_wCPU(Tensor *Tsr_In, int row_padsize, int col_padsize)
{
	assert(row_padsize > 0);
	assert(col_padsize > 0);
	
	Tsr_In->row += row_padsize;
	Tsr_In->col += col_padsize;
	
	for(int k = 0; k < Tsr_In->depth; k++)
	{
		// create bigger matrix based on the row- and col- padsize
		Tsr_In->vals[k] = realloc(Tsr_In->vals[k], Tsr_In->row * sizeof(float *));

		for(int i = 0; i < Tsr_In->row; i++)
		{
			Tsr_In->vals[k][i] = realloc(Tsr_In->vals[k][i], Tsr_In->col * sizeof(float));
		}
	
	}
}

Vector padding_2d_Vec_wCPU(Vector *Vec_In, int padsize)
{
	assert(padsize > 0);
	
	int templen = Vec_In->len + 2*padsize;
	
	Vector tempVec = create_vector(templen);
	
	for (int i = 0; i < tempVec.len; i++)
	{
		tempVec.vals[i + padsize] = Vec_In->vals[i]; 
	}
	
	return tempVec;
	
}

Matrix padding_2d_Mat_wCPU(Matrix *Mat_In, int padsize)
{
    assert(padsize > 0);
    
    int temprow = Mat_In->row + 2*padsize;
    int tempcol = Mat_In->col + 2*padsize;
    
    Matrix tempMat = create_matrix(temprow, tempcol);
    
    for (int i = 0; i < Mat_In->row; i++)
    {
        for (int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[i + padsize][j + padsize] = Mat_In->vals[i][j];
        }
    }
    
    return tempMat;
}

Tensor padding_2d_Tsr_wCPU(Tensor *Tsr_In, int padsize)
{
	assert(padsize > 0);
	
	int temprow = Tsr_In->row + 2*padsize;
    int tempcol = Tsr_In->col + 2*padsize;
    
    Tensor tempTsr = create_tensor(temprow, tempcol, Tsr_In->depth);
    
    for (int k = 0; k < Tsr_In->depth; k++)
    {
		for (int i = 0; i < Tsr_In->row; i++)
		{
			for (int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[k][i + padsize][j + padsize] = Tsr_In->vals[k][i][j];
			}
		}
	}

    return tempTsr;
}

void vpadding_2d_Mat_wCPU(Matrix *Mat_In, int padsize)
{
	assert(padsize > 0);
	
	int row_padsize = 2*padsize;
    int col_padsize = 2*padsize;
    
    Mat_In->row += row_padsize;
	Mat_In->col += col_padsize;
	
	// create bigger matrix based on the row- and col- padsize
	Mat_In->vals = realloc(Mat_In->vals, Mat_In->row * sizeof(float *));
	
	for(int i = 0; i < Mat_In->row; i++)
	{
		Mat_In->vals[i] = realloc(Mat_In->vals[i], Mat_In->col * sizeof(float));
	}
	
	// shift matrix values to the center
	for (int i = Mat_In->row - row_padsize; i >= 0; i--)
    {
		for (int j = Mat_In->col - col_padsize; j >= 0; j--)
		{
			Mat_In->vals[i + padsize][j + padsize] = Mat_In->vals[i][j];
		}
	}
	
	// clean up the top side rows by applying zero padding
	for(int i = 0; i < padsize; i++)
	{
		for(int j = 0; j < Mat_In->col; j++)
		{
			Mat_In->vals[i][j] = 0.0f;
		}
	}
	
	// clean up the left side columns by applying zero padding
	for(int i = 0; i < Mat_In->row; i++)
	{
		for(int j = 0; j < padsize; j++)
		{
			Mat_In->vals[i][j] = 0.0f;
		}
	}
}
  
void vpadding_2d_Tsr_wCPU(Tensor *Tsr_In, int padsize)
{
	assert(padsize > 0);
	
	int row_padsize = 2*padsize;
    int col_padsize = 2*padsize;
    
    Tsr_In->row += row_padsize;
	Tsr_In->col += col_padsize;
	
	for (int k = 0; k < Tsr_In->depth; k++)
	{
		// create bigger matrix at each layer based on the row- and col- padsize
		Tsr_In->vals[k] = realloc(Tsr_In->vals[k], Tsr_In->row * sizeof(float *));
		
		for(int i = 0; i < Tsr_In->row; i++)
		{
			Tsr_In->vals[k][i] = realloc(Tsr_In->vals[k][i], Tsr_In->col * sizeof(float));
		}
		
		// shift matrix values to the center
		for (int i = Tsr_In->row - row_padsize; i >= 0; i--)
		{
			for (int j = Tsr_In->col - col_padsize; j >= 0; j--)
			{
				Tsr_In->vals[k][i + padsize][j + padsize] = Tsr_In->vals[k][i][j];
			}
		}
		
		// clean up the top side rows by applying zero padding
		for(int i = 0; i < padsize; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				Tsr_In->vals[k][i][j] = 0.0f;
			}
		}
		
		// clean up the left side columns by applying zero padding
		for(int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < padsize; j++)
			{
				Tsr_In->vals[k][i][j] = 0.0f;
			}
		}

	}
}
