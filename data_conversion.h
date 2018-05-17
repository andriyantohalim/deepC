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
 * @file data_conversion.h
 * @brief Header file for data_conversion.h
 *
 * It is always necessary to be able to transform data from and to vector forms.
 * The following functions will be useful when fully-connected layers are part of 
 * the deep learning network.
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. parallelism
 * 
 * @bug No known bugs
 */
 
#ifndef DATA_CONVERSION_H
#define DATA_CONVERSION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

/**
 * @brief	Convert matrix to vector
 * @param 	Mat_In
 * @return 	Vector
 * 
 * This function converts matrix to vector
 */
Vector Mat2Vec_wCPU(Matrix *Mat_In);

/**
 * @brief	Convert tensor to vector
 * @param 	Tsr_In
 * @return 	Vector
 * 
 * This function converts tensor to vector
 */
Vector Tsr2Vec_wCPU(Tensor *Tsr_In);

/**
 * @brief	Convert vector to matrix
 * @param 	Vec_In
 * @param	Mat_Row
 * @param	Mat_Col
 * @return 	matrix
 * @note	vector's length must be the same to matrix's row*col
 * 
 * This function converts vector to matrix
 */
Matrix Vec2Mat_wCPU(Vector *Vec_In, int Mat_Row, int Mat_Col);

/**
 * @brief	Convert tensor to vector
 * @param 	Vec_In
 * @param	Tsr_Row
 * @param	Tsr_Col
 * @param	Tsr_Depth
 * @return 	Vector
 * @note	vector's length must be the same to tensor's row*col*depth
 * 
 * This function converts tensor to vector
 */
Tensor Vec2Tsr_wCPU(Vector *Vec_In, int Tsr_Row, int Tsr_Col, int Tsr_Depth);

#endif /* DATA_CONVERSION_H */
