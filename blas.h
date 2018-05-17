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
 * @file blas.h
 * @brief Header file for blas.c
 *
 * Basic Linear Algebra Subprograms (BLAS) consists of a set of low-level functions
 * for performing linear algebra operations.\n 
 * There are some other open-source implementation such as OpenBLAS, Armadillo, etc.
 * but it is not included here to remove dependencies and maintain datatype compatiblity.
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. Implement more BLAS functions
 * 2. implement OpenMP implementation for parallelism
 * 3. Implement various CPU architectures' SIMD instruction, i.e. AVX/AVX2 (x86), NEON (ARM)
 * 
 * @bug No known bugs
 * 
 * @see https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms
 */
 
#ifndef BLAS_H
#define BLAS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

/**
 * @brief	Scale vector by a factor
 * @param 	Vec_In
 * @param	scaling_factor
 * @return 	Vector
 * 
 * This function applies a scaling factor to each element of the input vector 
 * and return a new vector of the same dimension.
 */
Vector scale_Vec_wCPU(Vector *Vec_In, float scaling_factor);

/**
 * @brief	Scale matrix by a factor
 * @param 	Mat_In
 * @param	scaling_factor
 * @return 	Matrix
 * 
 * This function applies a scaling factor to each element of the input matrix 
 * and return a new matrix of the same dimension.
 */
Matrix scale_Mat_wCPU(Matrix *Mat_In, float scaling_factor);

/**
 * @brief	Scale tensor by a factor
 * @param 	Tsr_In
 * @param	scaling_factor
 * @return 	Tensor
 * 
 * This function applies a scaling factor to each element of the input tensor 
 * and return a new tensor of the same dimension.
 */
Tensor scale_Tsr_wCPU(Tensor *Tsr_In, float scaling_factor);

/**
 * @brief	Exponential operation on vector
 * @param 	Vec_In
 * @param	power
 * @return 	Matrix
 * 
 * This function applies an exponent to each element of the input vector 
 * and return a new vector of the same dimension.
 */
Vector power_Vec_wCPU(Vector *Vec_In, float power);

/**
 * @brief	Exponential operation on matrix
 * @param 	Mat_In
 * @param	power
 * @return 	Matrix
 * 
 * This function applies an exponent to each element of the input matrix 
 * and return a new matrix of the same dimension.
 */
Matrix power_Mat_wCPU(Matrix *Mat_In, float power);

/**
 * @brief	Exponential operation on tensor
 * @param 	Tsr_In
 * @param	power
 * @return 	Matrix
 * 
 * This function applies an exponent to each element of the input tensor
 * and return a new tensor of the same dimension.
 */
Tensor power_Tsr_wCPU(Tensor *Tsr_In, float power);

/**
 * @brief	Addition of two vectors
 * @param 	Vec_A
 * @param	Vec_B
 * @return 	Vector
 * @note 	Vectors dimension must be the same
 * 
 * This function adds two vectors, i.e. add each element of both vectors
 * and return a new vector of the same dimension.
 */
Vector addition_Vec_wCPU(Vector *Vec_A, Vector *Vec_B);

/**
 * @brief	Addition of two matrices
 * @param 	Mat_A
 * @param	Mat_B
 * @return 	Matrix
 * @note 	Matrices dimension must be the same
 * 
 * This function adds two matrices, i.e. add each element of both matrices
 * and return a new matrix of the same dimension.
 */
Matrix addition_Mat_wCPU(Matrix *Mat_A, Matrix *Mat_B);

/**
 * @brief	Addition of two tensors
 * @param 	Tsr_A
 * @param	Tsr_B
 * @return 	Tensor
 * @note 	Tensors dimension must be the same
 * 
 * This function adds two tensors, i.e. add each element of both tensors
 * and return a new tensor of the same dimension.
 */
Tensor addition_Tsr_wCPU(Tensor *Tsr_A, Tensor *Tsr_B);

/**
 * @brief	Dot product of two vectors
 * @param 	Vec_A
 * @param	Vec_B
 * @return 	float
 * @note 	Vectors dimension must be the same
 * @todo	
 * 1. To explore Kahan summation algorithm https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 * 2. To explore parallelism (OpenMP, CUDA, AVX/AVX2, NEON)
 * 
 * This function perform dot product on two vectors, i.e. element-wise 
 * multiplication and return a new vector of the same dimension.
 */
float dotproduct_Vec_wCPU(Vector *Vec_A, Vector *Vec_B);

/**
 * @brief	Dot product of two matrices
 * @param 	Mat_A
 * @param	Mat_B
 * @return 	float
 * @note 	Matrices dimension must be the same
 * @todo	
 * 1. To explore Kahan summation algorithm https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 * 2. To explore parallelism (OpenMP, CUDA, AVX/AVX2, NEON)
 * 
 * This function perform dot product on two matrices, i.e. element-wise 
 * multiplication and return a new vector of the same dimension.
 */
float dotproduct_Mat_wCPU(Matrix *Mat_A, Matrix *Mat_B);

/**
 * @brief	Dot product of two tensors
 * @param 	Tsr_A
 * @param	Tsr_B
 * @return 	float
 * @note 	Tensors dimension must be the same
 * @todo	
 * 1. To explore Kahan summation algorithm https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 * 2. To explore parallelism (OpenMP, CUDA, AVX/AVX2, NEON)
 * 
 * This function perform dot product on two tensors, i.e. element-wise 
 * multiplication and return a new tensor of the same dimension.
 */
float dotproduct_Tsr_wCPU(Tensor *Tsr_A, Tensor *Tsr_B);

/**
 * @brief	Multiply two vectors
 * @param 	Vec_A
 * @param	Vec_B
 * @return 	matrix
 * @todo	
 * 1. To explore parallelism (OpenMP, CUDA, AVX/AVX2, NEON)
 * 
 * This function perform multiplication on two vectors return a matrix.
 */
Matrix multiplication_Vec_wCPU(Vector *Vec_A, Vector *Vec_B);

/**
 * @brief	Matrix multiplication
 * @param 	Mat_A
 * @param	Mat_B
 * @return 	matrix
 * @todo	
 * 1. To explore parallelism (OpenMP, CUDA, AVX/AVX2, NEON)
 * 
 * Also known as "GEMM", General Matrix Multiplication.\n
 * This function perform multiplication on two matrices return a matrix
 */
Matrix multiplication_Mat_wCPU(Matrix *Mat_A, Matrix *Mat_B);

#endif /* BLAS_H */
