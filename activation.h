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
 * @file activation.h
 * @brief Header file for activation.c
 *
 * Activation functions define the output of a node given an input or set of inputs.\n
 * Usually activation functions are used to introduce non-linearity relationship
 * between input and output.
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. To implement more activation functions such as SELU
 * 2. To implement backprop (derivative) of each activation functions
 * 
 * @bug No known bugs
 * 
 * @see 
 * 1. https://en.wikipedia.org/wiki/Activation_function
 * 2. https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 * 3. https://en.wikipedia.org/wiki/Softmax_function
 * 4. https://en.wikipedia.org/wiki/Sigmoid_function
 */
 
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

/**
 * @brief	ReLU for vector
 * @param 	Vec_In
 * @return 	Vector
 * 
 * Rectified Linear Unit (ReLU) activation function for vector.\n
 * This function applies ReLU function on each element of input vector
 * and return a new vector of the same dimension.
 */
Vector ReLU_Vec_wCPU(Vector *Vec_In);

/**
 * @brief	ReLU for matrix
 * @param 	Mat_In
 * @return 	Matrix
 * 
 * Rectified Linear Unit (ReLU) activation function for matrix.\n
 * This function applies ReLU function on each element of input matrix
 * and return a new matrix of the same dimension.
 */
Matrix ReLU_Mat_wCPU(Matrix *Mat_In);

/**
 * @brief	ReLU for tensor
 * @param 	Tsr_In
 * @return 	Tensor
 * 
 * Rectified Linear Unit (ReLU) activation function for tensor.\n
 * This function applies ReLU function on each element of input tensor
 * and return a new tensor of the same dimension.
 */
Tensor ReLU_Tsr_wCPU(Tensor *Tsr_In);

/**
 * @brief	Leaky ReLU for vector
 * @param 	Vec_In
 * @return 	Vector
 * 
 * Leaky ReLU activation function for vector.\n
 * This function applies Leaky ReLU function on each element of input vector
 * and return a new vector of the same dimension.
 */
Vector Leaky_ReLU_Vec_wCPU(Vector *Vec_In);

/**
 * @brief	Leaky ReLU for matrix
 * @param 	Mat_In
 * @return 	Matrix
 * 
 * Leaky ReLU activation function for matrix.\n
 * This function applies Leaky ReLU function on each input matrix element.
 * and return a new matrix of the same dimension.
 */
Matrix Leaky_ReLU_Mat_wCPU(Matrix *Mat_In);

/**
 * @brief	Leaky ReLU for tensor
 * @param 	Tsr_In
 * @return 	Tensor
 * 
 * Leaky ReLU activation function for tensor.\n
 * This function applies Leaky ReLU function on each input tensor element.
 * and return a new tensor of the same dimension.
 */
Tensor Leaky_ReLU_Tsr_wCPU(Tensor *Tsr_In);

/**
 * @brief	Softmax for vector
 * @param 	Vec_In
 * @return 	Vector
 * 
 * Softmax activation function for vector.\n
 * This function applies softmax function on each element of input vector
 * and return a new vector of the same dimension.
 */
Vector softmax_Vec_wCPU(Vector *Vec_In);

/**
 * @brief	Softmax for matrix
 * @param 	Mat_In
 * @return 	Matrix
 * 
 * Softmax activation function for matrix.\n
 * This function applies softmax function on each element of input matrix
 * and return a new matrix of the same dimension.
 */
Matrix softmax_Mat_wCPU(Matrix *Mat_In);

/**
 * @brief	Softmax for tensor
 * @param 	Tsr_In
 * @return 	Tensor
 * 
 * Softmax activation function for vector.\n
 * This function applies softmax function on each element of input vector
 * and return a new tensor of the same dimension.
 */
Tensor softmax_Tsr_wCPU(Tensor *Tsr_In);

/**
 * @brief	Tanh for vector
 * @param 	Vec_In
 * @return 	Vector
 * 
 * Hyperbolic tangent activation function for vector.\n
 * This function applies tanh function on each element of input vector 
 * and return a new vector of the same dimension.
 */
Vector tanh_Vec_wCPU(Vector *Vec_In);

/**
 * @brief	Tanh for matrix
 * @param 	Mat_In
 * @return 	Matrix
 * 
 * Hyperbolic tangent activation function for matrix.\n
 * This function applies tanh function on each input elemnt of input matrix
 * and return a new matrix of the same dimension.
 */
Matrix tanh_Mat_wCPU(Matrix *Mat_In);

/**
 * @brief	Tanh for tensor
 * @param 	Tsr_In
 * @return 	Tensor
 * 
 * Hyperbolic tangent activation function for tensor.\n
 * This function applies tanh function on each element of input tensor
 * and return a new tensor of the same dimension.
 */
Tensor tanh_Tsr_wCPU(Tensor *Tsr_In);

/**
 * @brief	Sigmoid for vector
 * @param 	Vec_In
 * @return 	Vector
 * 
 * This function applies sigmoid function on each element of the input vector
 * and return a new vector of the same dimension.
 */
Vector sigmoid_Vec_wCPU(Vector *Vec_In);

/**
 * @brief	Sigmoid for matrix
 * @param 	Mat_In
 * @return 	Matrix
 * 
 * Sigmoid activation function for matrix.\n
 * This function applies sigmoid function on each element of the input matrix
 * and return a new matrix of the same dimension.
 */
Matrix sigmoid_Mat_wCPU(Matrix *Mat_In);

/**
 * @brief	Sigmoid for tensor
 * @param 	Tsr_In
 * @return 	Tensor
 * 
 * Sigmoid activation function for tensor.\n
 * This function applies sigmoid function on each element of the input tensor
 * and return a new tensor of the same dimension.
 */
Tensor sigmoid_Tsr_wCPU(Tensor *Tsr_In);

#endif /* ACTIVATION_H  */
