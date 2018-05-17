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
 * @file convolution.h
 * @brief Header file for convolution.c
 *
 * The core building block for deep learning on image/video.\n
 * In general, convolution can be performed in 1D (vector), 2D (matrix), or 3D (tensor).
 * In this implementation, only 2D convolutions are supported.
 * There are two type of convolutions, namely Valid Convolution and Same Convolution.
 * Valid Convolution produces smaller output matrix/tensors dimension due to the fact that 
 * there is no paddings are introduced.
 * Same Convolution produces matrix/tensors with the same dimension as the input due to the
 * use of paddings.
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. Parallelism: OpenMP implementation, CUDA kernel
 * 2. Backprop functions
 * 
 * @bug No known bugs
 * 
 * @see 
 * 1. https://en.wikipedia.org/wiki/Convolutional_neural_network
 * 2. https://stackoverflow.com/questions/42883547/what-do-you-mean-by-1d-2d-and-3d-convolutions-in-cnn
 */
 
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

#include "padding.h"

/**
 * @brief	Valid 2D convolution on matrix
 * @param 	Mat_In
 * @param 	Mat_kernel
 * @param 	stride
 * @return 	matrix
 * @note	stride value must be more than 0
 * 
 * This function performs 2D valid convolution on the input matrix 
 */
Matrix convolution_2d_Mat_wCPU(Matrix *Mat_In, Matrix *Mat_kernel, int stride);

/**
 * @brief	Same 2D convolution on matrix
 * @param 	Mat_In
 * @param	padsize
 * @param 	Mat_kernel
 * @param 	stride
 * @return 	matrix
 * @note	stride value must be more than 0
 * 
 * This function performs 2D same convolution on the input matrix 
 */
Matrix convolution_2d_with_pad_Mat_wCPU(Matrix *Mat_In, int padsize, Matrix *Mat_kernel, int stride);

/**
 * @brief	Valid 2D convolution on tensor
 * @param 	Tsr_In
 * @param 	Tsr_kernel
 * @param 	stride
 * @param	filter_size
 * @return 	Tensor
 * @note	stride value must be more than 0
 * 
 * This function performs 2D valid convolution on the input tensor 
 * and return a new tensor based on the specified filter size  
 */
Tensor convolution_2d_Tsr_wCPU(Tensor *Tsr_In, Tensor *Tsr_kernel, int stride, int filter_size);

/**
 * @brief	Same 2D convolution on tensor
 * @param 	Tsr_In
 * @param 	Tsr_kernel
 * @param 	stride
 * @param	filter_size
 * @return 	Tensor
 * @note	stride value must be more than 0
 * 
 * This function performs 2D same convolution on the input tensor 
 * and return a new tensor based on the specified filter size  
 */
Tensor convolution_2d_with_pad_Tsr_wCPU(Tensor *Tsr_In, int padsize, Tensor *Tsr_kernel, int stride, int filter_size);

#endif /* CONVOLUTION_H */
