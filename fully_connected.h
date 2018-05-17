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
 * @file fully_connected.h
 * @brief Header file for fully_connected.c
 *
 * Fully connected layer is usually placed at the end of the deep learning networks.
 * The last output of convolutional network is transformed into a vector and each element
 * is connected to a "neuron".\n
 * Fully connected layer can be used to construct a full-fledged artificial neural networks.
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. parallelism
 * 
 * @bug No known bugs
 * 
 * @see 
 * 1. https://en.wikipedia.org/wiki/Artificial_neural_network
 * 2. https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/fc_layer.html
 * 3. Backprop functions
 */
 
#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

/**
 * @brief	Fully-connected layer
 * @param 	Vec_In
 * @param	Mat_Weights
 * @param	Vec_Bias
 * @return 	Vector
 * @note	
 * 1. Weight matrix's column must be the same to input vector's length
 * 2. Weight matrix's row must be the same to bias vector's length
 * 3. Activation function is not included in this function and must be manually expressed
 * 
 * This function performs fully-connected layer
 */
Vector Fully_Connected_wCPU(Vector *Vec_In, Matrix *Mat_Weights, Vector *Vec_Bias);

#endif /* FULLY_CONNECTED_H */
