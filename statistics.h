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
 * @file statistics.h
 * @brief Header file for statistics.c
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
 
#ifndef STATISTICS_H
#define STATISTICS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

/**
 * @brief	Calculate mean of a vector
 * @param 	Vec_In
 * @return 	float
 * 
 * This function calculates and return the mean of a vector
 */
float mean_Vec_wCPU(Vector *Vec_In);

/**
 * @brief	Calculate mean of a vector
 * @param 	Mat_In
 * @return 	float
 * 
 * This function calculates and return the mean of a matrix
 */
float mean_Mat_wCPU(Matrix *Mat_In);

/**
 * @brief	Calculate mean of a tensor
 * @param 	Tsr_In
 * @return 	float
 * 
 * This function calculates and return the mean of a tensor
 */
float mean_Tsr_wCPU(Tensor *Tsr_In);

/**
 * @brief	Calculate variance of a vector
 * @param 	Vec_In
 * @return 	float
 * 
 * This function calculates and return the variance of a vector
 */
float variance_Vec_wCPU(Vector *Vec_In);

/**
 * @brief	Calculate variance of a matrix
 * @param 	Mat_In
 * @return 	float
 * 
 * This function calculates and return the variance of a matrix
 */
float variance_Mat_wCPU(Matrix *Mat_In);

/**
 * @brief	Calculate variance of a tensor
 * @param 	Tsr_In
 * @return 	float
 * 
 * This function calculates and return the variance of a tensor
 */
float variance_Tsr_wCPU(Tensor *Tsr_In);

/**
 * @brief	Calculate standard deviation of a vector
 * @param 	Vec_In
 * @return 	float
 * 
 * This function calculates and return the standard deviation of a vector
 */
float std_dev_Vec_wCPU(Vector *Vec_In);

/**
 * @brief	Calculate standard deviation of a matrix
 * @param 	Mat_In
 * @return 	float
 * 
 * This function calculates and return the standard deviation of a matrix
 */
float std_dev_Mat_wCPU(Matrix *Mat_In);

/**
 * @brief	Calculate standard deviation of a tensor
 * @param 	Tsr_In
 * @return 	float
 * 
 * This function calculates and return the standard deviation of a tensor
 */
float std_dev_Tsr_wCPU(Tensor *Tsr_In);

/**
 * @brief	Normalize vector
 * @param 	Vec_In
 * @return 	vector
 * 
 * This function normalize the input vector and return a normalized new vector
 * of the same dimension
 */
Vector normalization_Vec_wCPU(Vector *Vec_In);

/**
 * @brief	Normalize matrix
 * @param 	Mat_In
 * @return 	matrix
 * 
 * This function normalize the input matrix and return a normalized new matrix
 * of the same dimension
 */
Matrix normalization_Mat_wCPU(Matrix *Mat_In);

/**
 * @brief	Normalize tensor
 * @param 	Tsr_In
 * @return 	tensor
 * 
 * This function normalize the input tensor and return a normalized new tensor
 * of the same dimension
 */
Tensor normalization_Tsr_wCPU(Tensor *Tsr_In);

#endif /* STATISTICS_H */
