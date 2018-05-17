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
 * @file convolution.c
 * @brief Source file on detailed implementation for 2D convolution functions
 *
 * The core building block for deep learning networks on image/video.\n
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
 * 
 * @bug No known bugs
 * 
 * @see 
 * 1. https://en.wikipedia.org/wiki/Convolutional_neural_network
 * 2. https://stackoverflow.com/questions/42883547/what-do-you-mean-by-1d-2d-and-3d-convolutions-in-cnn
 */
 
#include "convolution.h"

Matrix convolution_2d_Mat_wCPU(Matrix *Mat_In, Matrix *Mat_kernel, int stride)
{
    assert(stride > 0);
    
    // Calculate output matrix size
    int temprow = (Mat_In->row - Mat_kernel->row)/stride + 1;
    int tempcol = (Mat_In->col - Mat_kernel->col)/stride + 1;
    
    // Calculate row- and col- boundaries for convolution to avoid rogue pointing 
    int rowbound = Mat_In->row - Mat_kernel->row + 1;
    int colbound = Mat_In->col - Mat_kernel->col + 1;
    
    // Create output matrix
    Matrix tempMat = create_matrix(temprow, tempcol);
    
    // p and q are the row- and col- indices for output matrix
	int p = 0;
	
    for(int i = 0; i < rowbound; i+= stride)
    {
        int q = 0; 
        
        for(int j = 0; j < colbound; j+= stride)
        {
            // Initialize a variable tempsum to store the summation of multiplication
            float tempsum = 0.0f;
            
            for(int m = 0; m < Mat_kernel->row; m++)
            {
                for(int n = 0; n < Mat_kernel->col; n++)
                {
                    tempsum += Mat_In->vals[i+m][j+n] * Mat_kernel->vals[m][n];
                }
            }
            
            // Assign tempsum to the respective output matrix component
            tempMat.vals[p][q] = tempsum;
 
            q++;
        }
        
        p++;
    }
    
    return tempMat;
}

Matrix convolution_2d_with_pad_Mat_wCPU(Matrix *Mat_In, int padsize, Matrix *Mat_kernel, int stride)
{
	assert(stride > 0);
	
	// Perform padding_2d
	vpadding_2d_Mat_wCPU(Mat_In, padsize);
	
	// Calculate output matrix size
	int temprow = (Mat_In->row - Mat_kernel->row)/stride + 1;
    int tempcol = (Mat_In->col - Mat_kernel->col)/stride + 1;
    
    // Calculate row- and col- boundaries for convolution to avoid rogue pointing
    int rowbound = Mat_In->row - Mat_kernel->row + 1;
    int colbound = Mat_In->col - Mat_kernel->col + 1;
      
    // Create output matrix  
    Matrix tempMat = create_matrix(temprow, tempcol);
    
    // p and q are the row- and col- indices for output matrix
    int p = 0;
    
    for(int i = 0; i < rowbound; i+= stride)
    {
		int q = 0;
		
		for(int j = 0; j < colbound; j+= stride)
		{
			float tempsum = 0.0f;
			
			for(int m = 0; m < Mat_kernel->row; m++)
			{
                for(int n = 0; n < Mat_kernel->col; n++)
                {
                    tempsum += Mat_In->vals[i+m][j+n] * Mat_kernel->vals[m][n];    
                }
            }
            
            // Assign tempsum to the respective output matrix component
            tempMat.vals[p][q] = tempsum;
            
            q++;
        }
        
        p++;
    }
    
    return tempMat;
}

Tensor convolution_2d_Tsr_wCPU(Tensor *Tsr_In, Tensor *Tsr_kernel, int stride, int filter_size)
{
	assert(stride > 0);
	assert(Tsr_In->depth == Tsr_kernel->depth);
	
	// Calculate output matrix size
    int temprow = (Tsr_In->row - Tsr_kernel->row)/stride + 1;
    int tempcol = (Tsr_In->col - Tsr_kernel->col)/stride + 1;
    
    // Calculate row- and col- boundaries for convolution to avoid rogue pointing 
    int rowbound = Tsr_In->row - Tsr_kernel->row + 1;
    int colbound = Tsr_In->col - Tsr_kernel->col + 1;
    
    // Create output tensor
    Tensor tempTsr = create_tensor(temprow, tempcol, filter_size);
    
    for(int k = 0; k < tempTsr.depth; k++)
    {	
		// p and q are the row- and col- indices for output matrix
		int p = 0;
	
		for(int i = 0; i < rowbound; i+= stride)
		{
			int q = 0; 
			
			for(int j = 0; j < colbound; j+= stride)
			{
				// Initialize a variable tempsum to store the summation of multiplication
				float tempsum = 0.0f;
				
				for (int o = 0; o < Tsr_kernel->depth; o++)
				// same effect: for(int o = 0; o < Tsr_In->depth; o++)
				{
					for(int m = 0; m < Tsr_kernel->row; m++)
					{
						for(int n = 0; n < Tsr_kernel->col; n++)
						{
							tempsum += Tsr_In->vals[o][i+m][j+n] * Tsr_kernel->vals[o][m][n];
						}
					}		
				}
				
				// Assign tempsum to the respective output matrix component
				tempTsr.vals[k][p][q] = tempsum;

				q++;
			}
			
			p++;
		}	
	}
    
    return tempTsr;
}

Tensor convolution_2d_with_pad_Tsr_wCPU(Tensor *Tsr_In, int padsize, Tensor *Tsr_kernel, int stride, int filter_size)
{
	assert(stride > 0);
	assert(Tsr_In->depth == Tsr_kernel->depth);

	// Perform padding_2d
	vpadding_2d_Tsr_wCPU(Tsr_In, padsize);
	
	// Calculate output matrix size
	int temprow = (Tsr_In->row - Tsr_kernel->row)/stride + 1;
    int tempcol = (Tsr_In->col - Tsr_kernel->col)/stride + 1;
    
    // Calculate row- and col- boundaries for convolution to avoid rogue pointing
    int rowbound = Tsr_In->row - Tsr_kernel->row + 1;
    int colbound = Tsr_In->col - Tsr_kernel->col + 1;
    
    // Create output tensor  
    Tensor tempTsr = create_tensor(temprow, tempcol, filter_size);
       
	for (int k = 0; k < filter_size; k++)
	{	
		// p and q are the row- and col- indices for output matrix
		int p = 0;
		
		for(int i = 0; i < rowbound; i+= stride)
		{
			int q = 0;
			
			for(int j = 0; j < colbound; j+= stride)
			{
				float tempsum = 0.0f;
				
				for (int o = 0; o < Tsr_kernel->depth; o++)
				// same effect: for(int o = 0; o < Tsr_In->depth; o++)
				{
					for(int m = 0; m < Tsr_kernel->row; m++)
					{
						for(int n = 0; n < Tsr_kernel->col; n++)
						{
							tempsum += Tsr_In->vals[o][i+m][j+n] * Tsr_kernel->vals[o][m][n];    
						}
					}
				}
				
				// Assign tempsum to the respective output matrix component
				tempTsr.vals[k][p][q] = tempsum;

				q++;
			}
			
			p++;
		}
	}		

    return tempTsr;
}
