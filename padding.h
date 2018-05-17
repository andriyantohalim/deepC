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
 * @file padding.h
 * @brief Header file for padding.c
 *
 * Paddings are usually needed for certain operations such as convolution and pooling.\n
 * In convolution, padding is introduced in the input matrix/tensor in order to perform
 * Same Convolution, i.e. output matrix/tensor dimension is the same to the input
 * matrix/tensor.\n
 * In pooling operation, it is possible for the pooling kernel to mismatch the 
 * input matrix/tensor dimension. Introducing some asymmetric padding will prevent the
 * rogue pointers from happening.
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo
 * 1. Change function name to indicate "symmetric"
 * 
 * @bug No known bugs
 * 
 * @see 
 * 1. https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
 */
 
#ifndef PADDING_H
#define PADDING_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

/**
 * @brief	Asymmetric padding on vector
 * @param 	Vec_In
 * @param 	padsize
 * @return 	vector
 * @note	Use this function when performing independent padding operation
 * 
 * This function pads the input vector after its last element and 
 * return a new vector
 */
Vector padding_asymmetric_Vec_wCPU(Vector *Vec_In, int padsize);

/**
 * @brief	Asymmetric padding on matrix
 * @param 	Mat_In
 * @param 	row_padsize
 * @param 	col_padsize
 * @return 	matrix
 * @note	Use this function when performing independent padding operation
 * 
 * This function pads the input matrix at the last element of each row, 
 * adds extra zero-padding rows at the bottom and return a new matrix
 */
Matrix padding_2d_asymmetric_Mat_wCPU(Matrix *Mat_In, int row_padsize, int col_padsize);

/**
 * @brief	Asymmetric padding on tensor
 * @param 	Tsr_In
 * @param 	row_padsize
 * @param 	col_padsize
 * @return 	tensor
 * @note	Use this function when performing independent padding operation
 * 
 * This function performs 2D zero-padding at each matrix layer of the input tensor except
 * the depth layer and return a new tensor.\n
 */
Tensor padding_2d_asymmetric_Tsr_wCPU(Tensor *Tsr_In, int row_padsize, int col_padsize);

/**
 * @brief	Asymmetric padding on vector
 * @param 	Vec_In
 * @param 	padsize
 * @return 	None
 * @note	Use this function when integrating padding with another function
 * 
 * This function pads the input vector after its last element and 
 * return a new vector
 */
void vpadding_asymmetric_Vec_wCPU(Vector *Vec_In, int padsize);

/**
 * @brief	Asymmetric padding on matrix
 * @param 	Mat_In
 * @param 	row_padsize
 * @param 	col_padsize
 * @return 	matrix
 * @note	Use this function when integrating padding with another function
 * 
 * This function pads the input matrix at the last element of each row, 
 * adds extra zero-padding rows at the bottom and return a new matrix
 */
void vpadding_2d_asymmetric_Mat_wCPU(Matrix *Mat_In, int row_padsize, int col_padsize);

/**
 * @brief	Asymmetric padding on tensor
 * @param 	Tsr_In
 * @param 	row_padsize
 * @param 	col_padsize
 * @return 	tensor
 * @note	Use this function when integrating padding with another function
 * 
 * This function performs 2D zero-padding at each matrix layer of the input tensor except
 * the depth layer and return a new tensor.\n
 */
void vpadding_2d_asymmetric_Tsr_wCPU(Tensor *Tsr_In, int row_padsize, int col_padsize);

/**
 * @brief	Symmetric padding on vector
 * @param 	Vec_In
 * @param 	padsize
 * @return 	vector
 * @note	Use this function when performing independent padding operation
 * 
 * This function pads the input vector before its first element and after its last element 
 * and return a new vector
 */
Vector padding_2d_Vec_wCPU(Vector *Vec_In, int padsize);

/**
 * @brief	Symmetric padding on matrix
 * @param 	Mat_In
 * @param 	padsize
 * @return 	matrix
 * @note	Use this function when performing independent padding operation
 * 
 * This function pads the input matrix at all of its sides and return a new matrix
 */
Matrix padding_2d_Mat_wCPU(Matrix *Mat_In, int padsize);

/**
 * @brief	Symmetric padding on tensor
 * @param 	Tsr_In
 * @param 	padsize
 * @return 	tensor
 * @note	Use this function when performing independent padding operation
 * 
 * This function pads the input tensor at all of its sides except its depth
 * and return a new tensor
 */
Tensor padding_2d_Tsr_wCPU(Tensor *Tsr_In, int padsize);

/**
 * @brief	Symmetric padding on matrix
 * @param 	Mat_In
 * @param 	padsize
 * @return 	None
 * @note	Use this function when integrating padding with another function
 * 
 * This function pads the input matrix at all of its sides and return a new matrix
 */
void vpadding_2d_Mat_wCPU(Matrix *Mat_In, int padsize);

/**
 * @brief	Symmetric padding on tensor
 * @param 	Tsr_In
 * @param 	padsize
 * @return 	None
 * @note	Use this function when integrating padding with another function
 * 
 * This function pads the input tensor at all of its sides except its depth
 * and return a new tensor
 */
void vpadding_2d_Tsr_wCPU(Tensor *Tsr_In, int padsize);

#endif /* PADDING_H */
