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
 * @file fully_connected.c
 * @brief Source file on detailed implementation for fully connected layer in Deep Learning
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
 
#include "fully_connected.h"

Vector Fully_Connected_wCPU(Vector *Vec_In, Matrix *Mat_Weights, Vector *Vec_Bias)
{
	assert(Mat_Weights->col == Vec_In->len);
	assert(Mat_Weights->row == Vec_Bias->len);

	Vector tempVec = create_vector(Mat_Weights->row);
	
	for(int i = 0; i < Mat_Weights->row; i++)
	{
		float tempval = 0.0f;
		
		for(int j = 0; j < Mat_Weights->col; j++)
		{
			tempval += Mat_Weights->vals[i][j] * Vec_In->vals[j];
		}
		
		tempVec.vals[i] = tempval + Vec_Bias->vals[i];
	}
	
	return tempVec;
}
