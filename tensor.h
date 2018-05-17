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
 * @file tensor.h
 * @brief Header file for tensor.c
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
 
#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

/**
 * @brief Define Tensor
 *
 * Define tensor data structure
 * 
 */
typedef struct Tensor
{
    int depth, row, col;
    float ***vals;
} Tensor;

Tensor create_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth);
Tensor constant_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth, float val);
Tensor random_tensor_normalized(int Tsr_Row, int Tsr_Col, int Tsr_Depth);
Tensor random_tensor_ranged(int Tsr_Row, int Tsr_Col, int Tsr_Depth, int Max_Val, int Min_Val);
void print_tensor(Tensor *Tsr);
void print_tensor_dim(Tensor *Tsr);
void free_tensor(Tensor *Tsr);
Tensor copy_Tsr_wCPU(Tensor *Tsr_In);

#endif /* TENSOR_H */
