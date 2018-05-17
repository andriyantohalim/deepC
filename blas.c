/****************************************************************************
 *                                                                          *
 * 	DeepC: Deep Learning/Machine Learning Inference Library written in C 	*
 * 																			*
 * 	Copyright (C) 2018 by Andriyanto Halim          						*
 *                                                                          *
 *  This program is free software: you can redistribute it and/or modify	*
 *  it under the terms of the GNU General Public License as published by	*
 *  the Free Software Foundation, either version 3 of the License, or		*
 *  (at your option) any later version.										*
 *                                                                          *
 *  This program is distributed in the hope that it will be useful,        	*
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of        	*
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          	*
 *  GNU Lesser General Public License for more details.                    	*
 *                                                                         	*
 *  You should have received a copy of the GNU Lesser General Public       	*
 *  License along with this program. If not, see							*
 *  <http://www.gnu.org/licenses/>. 										*
 * 																			*
 ****************************************************************************/
  
/**
 * @file blas.c
 * @brief Source file on detailed implementation for BLAS (Basic Linear Algebra Subprograms)
 *
 * Basic Linear Algebra Subprograms (BLAS) consists of a set of low-level functions
 * for performing linear algebra operations.\n 
 * There are some other open-source implementation such as OpenBLAS, Armadillo, etc.
 * but it is not included here to remove dependencies and maintain datatype compatiblity.
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. Implement more BLAS functions
 * 2. implement OpenMP implementation for parallelism
 * 3. Implement various CPU architectures' SIMD instruction, i.e. AVX/AVX2 (x86), NEON (ARM)
 * 
 * @bug No known bugs
 * 
 * @see https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
 */
 
#include "blas.h"

Vector scale_Vec_wCPU(Vector *Vec_In, float scaling_factor)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		tempVec.vals[i] = scaling_factor * Vec_In->vals[i];	
	}
	
	return tempVec;
}

Matrix scale_Mat_wCPU(Matrix *Mat_In, float scaling_factor)
{
    Matrix tempMat = create_matrix(Mat_In->row, Mat_In->col);

    for(int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[i][j] = scaling_factor * Mat_In->vals[i][j];
        }    
    }
    
    return tempMat;
}

Tensor scale_Tsr_wCPU(Tensor *Tsr_In, float scaling_factor)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);

    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for(int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[k][i][j] = scaling_factor * Tsr_In->vals[k][i][j];
			}    
		}
	}
    
    return tempTsr;
}

Vector power_Vec_wCPU(Vector *Vec_In, float power)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		tempVec.vals[i] = pow(Vec_In->vals[i], power);
	}
	
	return tempVec;
}

Matrix power_Mat_wCPU(Matrix *Mat_In, float power)
{
    Matrix tempMat = create_matrix(Mat_In->row, Mat_In->col);
      
    for(int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[i][j] = pow(Mat_In->vals[i][j], power);
        }    
    }
    
    return tempMat;
}

Tensor power_Tsr_wCPU(Tensor *Tsr_In, float power)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);
      
    for(int k = 0; k < Tsr_In->depth; k++)
    {  
		for(int i = 0; i < Tsr_In->row; i++)
		{
			for(int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[k][i][j] = pow(Tsr_In->vals[k][i][j], power);
			}    
		}
	}
    
    return tempTsr;
}

Vector addition_Vec_wCPU(Vector *Vec_A, Vector *Vec_B)
{
	assert(Vec_A->len == Vec_B->len);
	 
	Vector tempVec = create_vector(Vec_A->len);
	 
	for (int i = 0; i < Vec_A->len; i++)
	{
		tempVec.vals[i] = Vec_A->vals[i] + Vec_B->vals[i];
	}
	
	return tempVec; 
}

Matrix addition_Mat_wCPU(Matrix *Mat_A, Matrix *Mat_B)
{
    assert(Mat_A->row == Mat_B->row && Mat_B->col == Mat_B->col);
    
    Matrix tempMat = create_matrix(Mat_A->row, Mat_A->col);
    
    for(int i = 0; i < Mat_A->row; i++)
    {
        for(int j = 0; j < Mat_A->col; j++)
        {
            tempMat.vals[i][j] = Mat_A->vals[i][j] + Mat_B->vals[i][j];
        }
    }
    
    return tempMat;
}

Tensor addition_Tsr_wCPU(Tensor *Tsr_A, Tensor *Tsr_B)
{
    assert(Tsr_A->row == Tsr_B->row && Tsr_B->col == Tsr_B->col && Tsr_A->depth == Tsr_B->depth);
    
    Tensor tempTsr = create_tensor(Tsr_A->row, Tsr_A->col, Tsr_A->depth);
    
    for(int k = 0; k < Tsr_A->depth; k++)
    {
		for(int i = 0; i < Tsr_A->row; i++)
		{
			for(int j = 0; j < Tsr_A->col; j++)
			{
				tempTsr.vals[k][i][j] = Tsr_A->vals[k][i][j] + Tsr_B->vals[k][i][j];
			}
		}
	}
    
    return tempTsr;
}

float dotproduct_Vec_wCPU(Vector *Vec_A, Vector *Vec_B)
{
	assert(Vec_A->len == Vec_B->len);
	
	float tempsum = 0;
	
	for(int i = 0; i < Vec_A->len ; i++)
	{
		tempsum += Vec_A->vals[i] * Vec_B->vals[i];
	}
	
	return tempsum;	
}

float dotproduct_Mat_wCPU(Matrix *Mat_A, Matrix *Mat_B)
{
    assert(Mat_A->row == Mat_B->row && Mat_B->col == Mat_B->col);
    
    float tempsum = 0;
    
    for(int i = 0; i < Mat_A->row ; i++)
    {
        for(int j = 0; j < Mat_A->col ; j++)
        {
            tempsum += Mat_A->vals[i][j] * Mat_B->vals[i][j];
        }
    }
    
    return tempsum;   
}

float dotproduct_Tsr_wCPU(Tensor *Tsr_A, Tensor *Tsr_B)
{
    assert(Tsr_A->row == Tsr_B->row && Tsr_B->col == Tsr_B->col && Tsr_A->depth == Tsr_B->depth);
    
    float tempsum = 0;
    
    for(int k = 0; k < Tsr_A->depth; k++)
    {
		for(int i = 0; i < Tsr_A->row ; i++)
		{
			for(int j = 0; j < Tsr_A->col ; j++)
			{
				tempsum += Tsr_A->vals[k][i][j] * Tsr_B->vals[k][i][j];
			}
		}
	}
    
    return tempsum;   
}

Matrix multiplication_Vec_wCPU(Vector *Vec_A, Vector *Vec_B)
{
	Matrix tempMat = create_matrix(Vec_A->len, Vec_B->len);
	
	for(int i = 0; i < Vec_A->len; i++)
	{
		for(int j = 0; j < Vec_B->len; j++)
		{
			tempMat.vals[i][j] = Vec_A->vals[i] * Vec_B->vals[j];
		}
	}
	
	return tempMat;
}

Matrix multiplication_Mat_wCPU(Matrix *Mat_A, Matrix *Mat_B)
{
    assert(Mat_A->col == Mat_B->row);
    
    Matrix tempMat = create_matrix(Mat_A->row, Mat_B->col);
    
    // same effect: int tempcount = Mat_B->row;
    int tempcount = Mat_A->col; 
    
    for(int i = 0; i < Mat_A->row; i++)
    {
        for(int j = 0; j < Mat_B->col; j++)
        {
            for(int k = 0; k < tempcount; k++)
            {
                tempMat.vals[i][j] += Mat_A->vals[i][k] * Mat_B->vals[k][j];
            }
        }
    }
    
    return tempMat;
}

