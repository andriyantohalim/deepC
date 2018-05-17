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
 * @file vector.h
 * @brief Header file for vector.c
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
 
#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

/**
 * @brief Define Vector
 *
 * Define vector data structure
 * 
 */
typedef struct Vector
{
	int len;		/**< define vector length as int */
	float *vals;	/**< define vector values as float pointers */
} Vector;

/**
 * @brief	Function to create vector
 * @param 	Vec_Len
 * @return 	Vector
 * 
 * This function creates a vector and initialize all its value to 0.
 * The underlying implementation is based on calloc().
 */
Vector create_vector(int Vec_Len);

/**
 * @brief Function to create a constant vector
 * @param Vec_Len
 * @param val
 * @return Vector
 * 
 * This function creates a vector with values initialized to user-defined value.\n
 */
Vector constant_vector(int Vec_Len, float val);

/**
 * @brief Function to create a random normalized vector
 * @param Vec_Len
 * @return Vector
 * 
 * This function creates a random vector with a value normalized between 0.00 and 1.00\n
 */
Vector random_vector_normalized(int Vec_Len);

/**
 * @brief Function to create a random vector with specific range
 *
 * @param Vec_Len
 * @param Max_Val
 * @param Min_Val
 * @return Vector
 * 
 * This function creates a random vector with a value range specified by user.\n
 */
Vector random_vector_ranged(int Vec_Len, int Max_Val, int Min_Val);

/**
 * @brief Function to print vector
 * @param *Vec
 * @return None
 *
 * This function prints the vector values as well as its dimension
 */
void print_vector(Vector *Vec);

/**
 * @brief Function to print vector dimension
 * @param *Vec
 * @return None
 * 
 * This function prints only the vector dimension. Useful for quick glance
 * and troubleshooting.
 */
void print_vector_dim(Vector *Vec);

/**
 * @brief Function to free Vector
 * @param *Vec
 * @return None
 *
 * This function free the memory allocated for Vector data structure. 
 */
void free_vector(Vector *Vec);

/**
 * @brief Function to copy a Vector
 * @param *Vec_In
 * @return Vector 
 * 
 * This function returns a new vector identical to the Vec_In.
 *  
 */
Vector copy_Vec_wCPU(Vector *Vec_In);

#endif /* VECTOR_H */
