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
 * @file tensor.c
 * @brief Source file on detailed implementation for tensor related functions
 *
 * Collection of functions for tensor operations
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. change tensor length from "int" to "unsigned int"
 * 
 * @bug No known bugs
 * 
 * @see https://en.wikipedia.org/wiki/Array_data_structure
 */
 
#include "tensor.h"

Tensor create_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth)
{
	Tensor T;
	
	T.row = Tsr_Row;
	T.col = Tsr_Col;
	T.depth = Tsr_Depth;
	
	T.vals = calloc(T.depth, sizeof(float **));
	
	for (int k = 0; k < T.depth; k++)
	{
		T.vals[k] = calloc(T.row, sizeof(float *));
		
		for (int i = 0; i < T.row; i++)
		{
			T.vals[k][i] = calloc(T.col, sizeof(float));
		}
	}
	
	return T;
}

Tensor constant_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth, float val)
{
    Tensor tempTensor = create_tensor(Tsr_Row, Tsr_Col, Tsr_Depth);
    
    for (int k = 0; k < Tsr_Depth; k++)
    {
		for (int i = 0; i < Tsr_Row; i++)
		{
			for (int j = 0; j < Tsr_Col; j++)
			{
				tempTensor.vals[k][i][j] = val;
			}
		}
	}
    
    return tempTensor;
}

Tensor random_tensor_normalized(int Tsr_Row, int Tsr_Col, int Tsr_Depth)
{
    Tensor tempTensor = create_tensor(Tsr_Row, Tsr_Col, Tsr_Depth);
    
    for (int k = 0; k < Tsr_Depth; k++)
    {
		for (int i = 0; i < Tsr_Row; i++)
		{
			for (int j = 0; j < Tsr_Col; j++)
			{
				tempTensor.vals[k][i][j] = (float)(rand())/RAND_MAX;
			}
		}
	}
    
    return tempTensor;
}
 
Tensor random_tensor_ranged(int Tsr_Row, int Tsr_Col, int Tsr_Depth, int Max_Val, int Min_Val)
{
    Tensor tempTensor = create_tensor(Tsr_Row, Tsr_Col, Tsr_Depth);
    
    for (int k = 0; k < Tsr_Depth; k++)
    {
		for (int i = 0; i < Tsr_Row; i++)
		{
			for (int j = 0; j < Tsr_Col; j++)
			{
				tempTensor.vals[k][i][j] = (float)(((rand()) % (Max_Val - Min_Val + 1)) + Min_Val);
			}
		}
	}
    
    return tempTensor;
}

void print_tensor(Tensor *Tsr)
{
	printf("Matrix dim: %d x %d\n", Tsr->row, Tsr->col);
	printf("Layer size: %d\n", Tsr->depth);

	for(int k = 0; k < Tsr->depth; ++k)
	{
		printf("Matrix[%d]:\n", k);
		
		for (int i = 0; i < Tsr->row; ++i)
		{
			for(int j = 0; j < Tsr->col; ++j)
			{
				printf("%.2f\t", Tsr->vals[k][i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	printf("\n\n");
}

void print_tensor_dim(Tensor *Tsr)
{
	printf("Matrix dim: %d x %d\n", Tsr->row, Tsr->col);
	printf("Layer size: %d\n", Tsr->depth);
	printf("\n\n");
}

void free_tensor(Tensor *Tsr)
{
    for(int k = 0; k < Tsr->depth; k++)
    {
		for(int i = 0; i < Tsr->row; i++)
		{
			free(Tsr->vals[k][i]);
		}
		
		free(Tsr->vals[k]);
	}

    free(Tsr->vals);
}

Tensor copy_Tsr_wCPU(Tensor *Tsr_In)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for (int i = 0; i < Tsr_In->row; i++)
		{
			for (int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[k][i][j] = Tsr_In->vals[k][i][j];
			}
		}
	}
		
    return tempTsr;
}

