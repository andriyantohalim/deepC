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
 * @file main.c
 * @brief Main function, where the magic takes place
 *
 * Here typically goes a more extensive explanation of what the header
 * defines. Doxygens tags are words preceeded by either a backslash @\
 * or by an at symbol @@.
 * 
 * @see http://www.stack.nl/~dimitri/doxygen/docblocks.html
 * @see http://www.stack.nl/~dimitri/doxygen/commands.html
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @bug No known bugs
 */
 
#include "main.h"
#include <time.h>

// Compilation and run command
// gcc *.c -std=c99 -o main -lm && ./main

float sigmoid_Scl_wCPU(float Scl_In)
{
	float tempScl = 1.0/(1.0 + exp(-(Scl_In))); 
	return tempScl;
}

int main(void)
{    	 
	float X[3] = {-0.0002075f, -0.00021f, 0.020034f};
	float Y = 1.0f;
	
	// Initialize Weights and bias
	float W[3] = {0};
	float B = 0.000115f;
	
	for(long int epoch = 0; epoch < 20000; epoch++)
	{
		printf("=============== EPOCH #%ld =============== \n", epoch);
		
		// STEP 1. Forward Propagation
		float tempval = 0.0f;
		
		for (int i = 0; i < 3; i++)
		{
			tempval += W[i]*X[i];
		}
		
		float A0 = tempval + B;
		float A1 = sigmoid_Scl_wCPU(A0);
		//~ printf("A1 = %f\tY = %f\n", A1, Y);
		
		// STEP 2. Calculate Error
		float errorval = A1 - Y;
		printf("A1 = %f\tY = %f\terror = %f\n", A1, Y, errorval);
		
		// STEP 3. Backward Propagation
		tempval = 0.0f;
		
		for (int i = 0; i < 3; i++)
		{
			tempval += errorval*X[i];
		}
		
		float dW = tempval/3;
		
		tempval = 0.0f;
		
		for (int i = 0; i < 3; i++)
		{
			tempval += errorval;
		}
		
		float dB = tempval/3;
		
		printf("dW = %f, dB = %f\n", dW, dB);
		
		// STEP 4. Update Weights and Bias
		float learning_rate = 0.0001f;
		
		for(int i = 0; i < 3; i++)
		{
			W[i] = W[i] - learning_rate*dW;
			printf("W[%d] = %f\t",i, W[i]);
		}
		printf("\n");
		
		B = B - learning_rate*dB;
		printf("B = %f\n", B);
		
		
		// STEP 5. Calculate Cost Function
		float tempcost = 0.0f;
		float cost = 0.0f;
		
		tempcost = -(Y * log(A1) + (1 - Y) * log(1 - A1));

		cost = tempcost;
		printf("Cost = %f\n", cost);
		
		printf("\n");
	}
	
	return 0;
}
