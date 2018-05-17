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
 * @file matrix.h
 * @brief Header file for matrix.c
 *
 * Collection of functions for matrix operations
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. change matrix length from "int" to "unsigned int"
 * 
 * @bug No known bugs
 * 
 * @see https://en.wikipedia.org/wiki/Array_data_structure
 */
 
#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

/**
 * @brief Define Matrix
 *
 * Define Matrix data structure
 * 
 */
typedef struct Matrix
{
    int row, col;
    float **vals;
} Matrix;


Matrix create_matrix(int Mat_Row, int Mat_Col);
Matrix constant_matrix(int Mat_Row, int Mat_Col, float val);
Matrix random_matrix_normalized(int Mat_Row, int Mat_Col);
Matrix random_matrix_ranged(int Mat_Row, int Mat_Col, int Max_Val, int Min_Val);
void print_matrix(Matrix *Mat);
void print_matrix_dim(Matrix *Mat);
void free_matrix(Matrix *Mat);
Matrix copy_Mat_wCPU(Matrix *Mat_In);
Matrix transpose_Mat_wCPU(Matrix *Mat_In);

#endif /* MATRIX_H */
