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
 * @file vector.c
 * @brief Source file on detailed implementation for vector related functions
 *
 * Collection of functions for vector operations
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. change vector length from "int" to "unsigned int"
 * 
 * @bug No known bugs
 * 
 * @see https://en.wikipedia.org/wiki/Array_data_structure
 */
 
#include "vector.h"

Vector create_vector(int Vec_Len)
{
	Vector V; 
	
	V.len = Vec_Len;  
	
	V.vals = calloc(V.len, sizeof(float));
	
	return V;
}

Vector constant_vector(int Vec_Len, float val)
{
	Vector tempVec = create_vector(Vec_Len);
	
	for (int i = 0; i < Vec_Len; i++)
	{
		tempVec.vals[i] = val;
	}
	
	return tempVec;
}

Vector random_vector_normalized(int Vec_Len)
{
	Vector tempVec = create_vector(Vec_Len);
	
	srand(time(NULL));
	
	for (int i = 0; i < Vec_Len; i++)
	{
		tempVec.vals[i] = (float)(rand())/RAND_MAX;
	}
	
	return tempVec;
}

Vector random_vector_ranged(int Vec_Len, int Max_Val, int Min_Val)
{
	Vector tempVec = create_vector(Vec_Len);
	
	srand(time(NULL));
	
	for (int i = 0; i < Vec_Len; i++)
	{
		tempVec.vals[i] = (float)(((rand()) % (Max_Val - Min_Val + 1)) + Min_Val);
	}
	
	return tempVec;
}

void print_vector(Vector *Vec)
{
	printf("Len: %d\n", Vec->len);
	
	for (int i = 0; i < Vec->len; i++)
	{
		printf("%.2f\t", Vec->vals[i]);
	}
	
	printf("\n\n");
}

void print_vector_dim(Vector *Vec)
{
	printf("Len: %d \n", Vec->len);
    printf("\n\n");
}

void free_vector(Vector *Vec)
{
	free(Vec->vals);
}

Vector copy_Vec_wCPU(Vector *Vec_In)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		tempVec.vals[i] = Vec_In->vals[i];
	}
	
	return tempVec;
}






