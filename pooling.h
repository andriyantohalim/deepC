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
 * @file pooling.h
 * @brief Header file for pooling.c
 *
 * Pooling is a method of downsampling the layer in a non-linear fashion.\n
 * Only max-pooling is supported in this implementation as it is the most 
 * common one found in deep learning networks.
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo
 * 1. Parallelism
 * 2. backprop functions
 * 
 * @bug No known bugs
 * 
 * @see
 * 1. https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer
 */
 
#ifndef POOLING_H
#define POOLING_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

#include "padding.h"

/**
 * @brief	Maxpooling on matrix
 * @param 	Mat_In
 * @param	filter_height
 * @param	filter_width
 * @param 	stride
 * @return 	matrix
 * @note	
 * 1. stride value must be more than 0
 * 2. filter height or filter width must be the same to the stride value
 * 
 * This function performs maxpooling on a matrix and return a smaller output matrix
 */
Matrix maxpooling_Mat_wCPU(Matrix *Mat_In, int filter_height, int filter_width, int stride);

/**
 * @brief	Maxpooling on tensor
 * @param 	Tsr_In
 * @param	filter_height
 * @param	filter_width
 * @param 	stride
 * @return 	tensor
 * @note	
 * 1. stride value must be more than 0
 * 2. filter height or filter width must be the same to the stride value
 * 
 * This function performs maxpooling on each layer of a tensor and return a smaller output tensor
 */
Tensor maxpooling_Tsr_wCPU(Tensor *Tsr_In, int filter_height, int filter_width, int stride);

#endif /* POOLING_H */
