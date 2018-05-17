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
 * @file statistics.c
 * @brief Source file on detailed implementation for statistics functions
 *
 * Some statistical operations such as means, variance and standard deviation are 
 * needed in deep learning/machine learning processes in order to perform normalization.
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. To implement more statistics functions relevant to deep learning/machine learning
 * 
 * @bug No known bugs
 * 
 * @see 
 * 1. https://en.wikipedia.org/wiki/Mean
 * 2. https://en.wikipedia.org/wiki/Standard_deviation
 * 3. https://en.wikipedia.org/wiki/Normalization_(statistics)
 */
 
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
